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
from utils.dataset_loaders import data2dataset_OOV
from utils.metrics import nltk_bleu
from utils.beam_search import beam_search_decode

from models_and_trainers.BERT_style_modules import BERTStyleEncoder, BERTStyleDecoder
from models_and_trainers.exposed_transformer import Transformer

class CopyGeneratorModel():
    def __init__(self, vocab, vocab_size, embed_dim, att_heads, layers, dim_feedforward, lr, max_seq_length, use_copy=True):
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CopyGeneratorTransformer(vocab_size, 
                                             embed_dim, 
                                             att_heads, 
                                             layers, 
                                             dim_feedforward, 
                                             dropout=0.5, 
                                             use_copy=True)
        self.model.to(self.device)
        self.max_seq_length = max_seq_length
        self.use_copy = use_copy
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.99)
        self.data2dataset_fn = data2dataset_OOV
        self.model_name = f"copy_gen_vcb{vocab_size}"
        
        
    def train_step(self, batch, metadata):
        self.model.train() # Turn on the train mode
        total_loss = 0.
        start_time = time.time()
        tgt_vocab_size = len(self.vocab.itos) + self.max_seq_length
        encoder_input = batch.src
        decoder_input = batch.tgt[:-1]
        targets = batch.tgt[1:]

        self.optimizer.zero_grad()
        output = self.model(encoder_input, decoder_input)
        
        criterion = nn.CrossEntropyLoss(ignore_index=self.vocab.stoi['<pad>'])
        loss = criterion(output.view(-1, tgt_vocab_size), targets.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()
        elapsed = time.time() - start_time
        return loss
    
    
    def batch_filter_ids(self, batch_list):
        return [[id for id in l if id not in [1,2,3]] for l in batch_list]
    
    def evaluate_iterator(self, iterator, metadata):
        BLEU_scores = []
        log = False
        with open(f"{self.model_name}_eval.out", "w", encoding="utf-8") as out_fp:
            for i, batch in enumerate(iterator):
                batch_size = batch.src.shape[1]

                encoder_inputs = batch.src
                predictions = beam_search_decode(self.model,
                                  batch_encoder_ids=encoder_inputs,
                                  SOS_token=self.vocab.stoi["<sos>"],
                                  EOS_token=self.vocab.stoi["<eos>"],
                                  PAD_token=self.vocab.stoi["<pad>"],
                                  beam_size=1,
                                  max_length=30,
                                  num_out=1)

                sources = encoder_inputs.transpose(0,1).cpu().tolist()
                sources = self.batch_filter_ids(sources)

                predictions = [t[0].view(-1).cpu().tolist() for t in predictions]
                predictions = self.batch_filter_ids(predictions)

                targets = batch.tgt.transpose(0,1).cpu().tolist()
                targets = self.batch_filter_ids(targets)

    #             print(batch.tgt)

                OOVss = [[metadata.OOV_itos[OOV] for OOV in batch.OOVs.cpu()[:,idx].tolist() if OOV != 3] for idx in range(batch_size)]

                if i % int(len(iterator)/3) == 0:
                    print("| EVALUATION | {:5d}/{:5d} batches |".format(i, len(iterator)))

                for j in range(batch_size):
                    BLEU = nltk_bleu(targets[j], predictions[j])
                    BLEU_scores.append(BLEU)

                    out_fp.write("SRC  :" + self.vocab.decode(sources[j], OOVss[j]) + "\n")
                    out_fp.write("TGT  :" + self.vocab.decode(targets[j], OOVss[j]) + "\n")
                    out_fp.write("PRED :" + self.vocab.decode(predictions[j], OOVss[j]) + "\n")
                    out_fp.write("BLEU :" + str(BLEU) + "\n")
                    out_fp.write("\n")

                    if log:
                        print("SRC  :" + self.vocab.decode(sources[j], OOVss[j]))
                        print("TGT  :" + self.vocab.decode(targets[j], OOVss[j]))
                        print("PRED :" + self.vocab.decode(predictions[j], OOVss[j]))
                        print("BLEU :" + str(BLEU))
                        print()
            out_fp.write("\n\n| EVALUATION | BLEU: {:5.2f} |\n".format(np.average(BLEU_scores)))
#             print("| EVALUATION | BLEU: {:5.3f} |".format(np.average(BLEU_scores)))
            return np.average(BLEU_scores)
        
        
    def save(save_fp):
        torch.save((self.model, self.optimizer, self.scheduler, self.vocab), save_fp)
    
    def restore(restore_fp):
        self.model, self.optimizer, self.scheduler, self.vocab = torch.load(restore_fp)


class CopyGeneratorTransformer(nn.Module):

    def __init__(self, vocab_size=30522, embed_dim=768, att_heads=12, layers=12, dim_feedforward=3072, dropout=0.1, use_copy=True, masked_look_ahead_att=True, max_seq_length=200):
        super(CopyGeneratorTransformer, self).__init__()
        self.model_type = 'Transformer'
        
        self.use_copy = use_copy
        self.masked_look_ahead_att = masked_look_ahead_att
        
        self.embedding_size = embed_dim
        self.bert_encoder_model = BERTStyleEncoder(vocab_size=vocab_size, dim_model=embed_dim, nhead=att_heads, \
                 num_encoder_layers=layers, d_feed=dim_feedforward, dropout=dropout)
#         self.bert_encoder_model.load_pretrained()
        
        self.src_embedder = self.bert_encoder_model.embedder
        self.src_encoder = self.bert_encoder_model.encoder
        
        self.bert_decoder_model = BERTStyleDecoder(vocab_size=vocab_size, dim_model=embed_dim, nhead=att_heads, \
                 num_encoder_layers=layers, d_feed=dim_feedforward, dropout=dropout)
#         self.bert_decoder_model.load_pretrained()
        
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.att_mask_noise = 0.0
        
        self.mask_index = [None] + [self._generate_square_subsequent_mask(i).to(self.device) for i in range(1,max_seq_length)]
        
        if use_copy:
            self.p_generator = nn.Linear(embed_dim,1)

        self.init_weights()
        self.tgt_mask = None

    def init_weights(self):
        initrange = 0.1
#         self.src_embedder.weight.data.uniform_(-initrange, initrange)
#         self.tgt_embedder.weight.data.uniform_(-initrange, initrange)
        
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
        
    def _generate_square_subsequent_mask(self, sz):
#         noise_e = 0.05 if self.training else 0.0 # this is code to add noise to the decoding process during training
        noise_e = self.att_mask_noise if self.training else 0.0
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

    def forward(self, src, tgt):
        self.tgt_mask = self._generate_square_subsequent_mask(len(tgt)).to(self.device) if self.masked_look_ahead_att else None
        

        src_emb = self.src_embedder(src) * math.sqrt(self.embedding_size)
        
        tgt_emb = self.tgt_embedder(tgt) * math.sqrt(self.embedding_size)
        
        output, atts = self.transformer(src_emb, tgt_emb, tgt_mask=self.tgt_mask)
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

