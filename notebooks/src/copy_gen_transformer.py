import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torchtext.data import Field, BucketIterator
import torchtext
import time
import math
import numpy as np

from IPython.core.debugger import set_trace as tr
from .base_transformer import TransformerModel, PositionalEncoding
from dotmap import DotMap
import sys
sys.path.append("..") # this will add the previous directory to the ppython path allowing for subdirectory imports

from .BERT_style_modules import BERTStyleEncoder, BERTStyleDecoder
from .exposed_transformer import Transformer


class CopyGeneratorTransformer(nn.Module):

    def __init__(self, vocab_size=30522, embed_dim=768, att_heads=12, layers=12, dim_feedforward=3072, dropout=0.1, use_copy=True, \
                 masked_look_ahead_att=True, pretrained_encoder=False, pretrained_decoder=False):
        super().__init__()
        self.model_type = 'Transformer'
        
        self.use_copy = use_copy
        self.masked_look_ahead_att = masked_look_ahead_att
        
        self.embedding_size = embed_dim
        self.bert_encoder_model = BERTStyleEncoder(vocab_size=vocab_size, dim_model=embed_dim, nhead=att_heads, \
                 num_encoder_layers=layers, d_feed=dim_feedforward, dropout=dropout)
        if pretrained_encoder or pretrained_decoder:
            assert vocab_size==30522 and embed_dim==768 and att_heads==12 and layers==12 and dim_feedforward==3072, f"Model architecture doesn't match pretrained model,vocab_size {vocab_size}, embed_dim {embed_dim}, att_heads {att_heads}, layers {layers}, dim_feedforward {dim_feedforward}"
        if pretrained_encoder:
            self.bert_encoder_model.load_pretrained()
        
        self.src_embedder = self.bert_encoder_model.embedder
        self.src_encoder = self.bert_encoder_model.encoder
        
        self.bert_decoder_model = BERTStyleDecoder(vocab_size=vocab_size, dim_model=embed_dim, nhead=att_heads, \
                 num_encoder_layers=layers, d_feed=dim_feedforward, dropout=dropout)
        if pretrained_decoder:
            self.bert_decoder_model.load_pretrained()
        
        self.tgt_embedder = self.bert_encoder_model.embedder # nn.Embedding(vocab_size, embed_dim)
        self.tgt_decoder = self.bert_decoder_model.decoder
        
        self.transformer = Transformer(d_model=embed_dim,
                                       nhead=att_heads, 
                                       num_encoder_layers=layers, 
                                       num_decoder_layers=layers, 
                                       dim_feedforward=dim_feedforward,
                                       custom_encoder=self.src_encoder,
                                       custom_decoder=self.tgt_decoder)
        self.decoder = nn.Linear(embed_dim, vocab_size)
        self.att_mask_noise = 0.0
                
        if use_copy:
            self.p_generator = nn.Linear(embed_dim,1)

        self.init_weights()
        self.tgt_mask = None
    
    @property
    def device(self):
        return self.src_embedder.word_embeddings.weight.device

    def init_weights(self):
        initrange = 0.1
#         self.src_embedder.weight.data.uniform_(-initrange, initrange)
#         self.tgt_embedder.weight.data.uniform_(-initrange, initrange)
        
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None):
        self.tgt_mask = self.transformer.generate_square_subsequent_mask(len(tgt)).to(self.device) if self.masked_look_ahead_att else None

        src_emb = self.src_embedder(src) * math.sqrt(self.embedding_size)
        
        tgt_emb = self.tgt_embedder(tgt) * math.sqrt(self.embedding_size)
        
        output, atts = self.transformer(src_emb, tgt_emb, tgt_mask=self.tgt_mask, \
                                        src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask, \
                                        memory_key_padding_mask=src_key_padding_mask)
        vocab_output = self.decoder(output)
        
        if self.use_copy:
            src_scat = src.transpose(0,1)
            src_scat = src_scat.unsqueeze(0)
            src_scat = torch.repeat_interleave(src_scat, tgt.shape[0], dim=0)
    #         print("src_scat.shqape", src_scat.shape)

            p_gens = self.p_generator(output).sigmoid()
            atts = atts.transpose(0,1)
    #         print("att.shqape", atts.shape)
            atts = atts * (1 - p_gens)

#             output = self.decoder(output)
    #         output[:,:,12:] = -np.inf
            vocab_output = vocab_output.softmax(-1)
            vocab_output = vocab_output * p_gens

            vocab_output = vocab_output.scatter_add_(2,src_scat,atts)
            vocab_output = vocab_output.log()
        
        return vocab_output

