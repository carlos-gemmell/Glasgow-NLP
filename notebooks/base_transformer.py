import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchtext
import tqdm
from torchtext.data import Field, BucketIterator

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    

class TransformerModel(nn.Module):

    def __init__(self, src_vocab_size, tgt_vocab_size, embedding_size=512, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        
        self.embedding_size = embedding_size
        self.pos_encoder = PositionalEncoding(embedding_size, dropout)
        self.src_encoder = nn.Embedding(src_vocab_size, embedding_size)
        self.tgt_encoder = nn.Embedding(tgt_vocab_size, embedding_size)
        
        self.transformer = nn.Transformer(d_model=embedding_size, nhead=8, num_encoder_layers=4, num_decoder_layers=4, dim_feedforward=1024)
        self.decoder = nn.Linear(embedding_size, tgt_vocab_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.init_weights()
        self.tgt_mask = None

    def _generate_square_subsequent_mask(self, sz):
#         noise_e = 0.05 if self.training else 0.0 # this is code to add noise to the decoding process during training
        noise_e = 0.0 if self.training else 0.0
        noise_mask = (torch.rand(sz,sz) > noise_e).float()

        mask = (torch.triu(torch.ones(sz,sz))).transpose(0, 1)
        mask = torch.mul(mask, noise_mask)
        v = (torch.sum(mask, dim=-1) == 0).float()

        fix_mask = torch.zeros(sz,sz)
        fix_mask[:,0] = 1.0
        v = v.repeat(sz, 1).transpose(0,1)
        fix_mask = torch.mul(fix_mask,v)

        mask += fix_mask
        
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.src_encoder.weight.data.uniform_(-initrange, initrange)
        self.tgt_encoder.weight.data.uniform_(-initrange, initrange)
        
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, tgt):
        self.tgt_mask = self._generate_square_subsequent_mask(len(tgt)).to(self.device)

        src = self.src_encoder(src) * math.sqrt(self.embedding_size)
        src = self.pos_encoder(src)
        
        tgt = self.tgt_encoder(tgt) * math.sqrt(self.embedding_size)
        tgt = self.pos_encoder(tgt)
        
        output = self.transformer(src, tgt, tgt_mask=self.tgt_mask)
        output = self.decoder(output)
        return output