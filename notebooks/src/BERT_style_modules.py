from transformers import BertModel, BertForMaskedLM, BertTokenizer
from .bert_pretrain_convert.state_dict_translate import translate_from_hugginface_to_torch_BERT
from torch.nn.modules import LayerNorm

import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from .exposed_transformer import TransformerDecoder, TransformerDecoderLayer
import json
import os

class BERTStyleEmbedding(nn.Module):

    def __init__(self, d_model, vocab_size=30522, dropout=0.1, max_len=512):
        super(BERTStyleEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        self.word_embeddings = nn.Embedding(vocab_size,d_model)
        self.position_embeddings = nn.Embedding(max_len, d_model)
        self.token_type_embeddings = nn.Embedding(2, d_model)
        
        self.embedding_layer_norm = LayerNorm(d_model, eps=1e-12)
        
    def forward(self, input_ids):
        
        input_shape = input_ids.shape
        seq_length = input_shape[0]

        position_ids = torch.arange(seq_length, dtype=torch.long, device=self.get_device())
        position_ids = position_ids.unsqueeze(1).expand(input_shape)

        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.get_device())
        
        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = word_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.embedding_layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    
    def get_device(self):
        return next(self.parameters()).device
    
    
    
    
class BERTStyleEncoder(nn.Module):
    
    def __init__(self, vocab_size=30522, dim_model=768, nhead=12, name="music_bert",\
                 num_encoder_layers=12, d_feed=3072, dropout=0.1):
        super(BERTStyleEncoder, self).__init__()
        self.vocab_size = vocab_size
        
        self.embedder = BERTStyleEmbedding(dim_model, self.vocab_size)
        
        encoder_layer = TransformerEncoderLayer(dim_model, nhead, \
                                                dim_feedforward=d_feed, \
                                                dropout=dropout,
                                                activation='gelu')
        
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
    
    def forward(self, src_tokens):
        
        sequence = self.embedder(src_tokens)
        output = self.encoder(sequence)
        
        return output
    
    def get_device(self):
        return next(self.parameters()).device
        
    
    def load_pretrained(self):
        #load pretrained model from Hugginface
        
        device = self.get_device()
        bertModel = BertModel.from_pretrained("bert-base-uncased").to(device)
        script_dir = os.path.dirname(__file__)
        with open(os.path.join(script_dir, "layer_maps/bert_encoder_extras_map.json"), "r") as f:
            extras_map = json.load(f)
        with open(os.path.join(script_dir, "layer_maps/bert_encoder_layer_map.json"), "r") as f:
            layer_map = json.load(f)
        with open(os.path.join(script_dir, "layer_maps/bert_encoder_self_attention_map.json"), "r") as f:
            self_att_map = json.load(f)
        
        bert_state_dict = translate_from_hugginface_to_torch_BERT(bertModel.state_dict(), extras_map, layer_map, self_att_map)
        unset_keys = self.load_state_dict(bert_state_dict, strict=False)
        
    
        print(unset_keys)
        self.__test_integrity()
        
        
    def __test_integrity(self):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        BERTmodel = BertModel.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("this is a test")).unsqueeze(0)
        BERT_outputs = BERTmodel(input_ids)[0]
        self.eval()
        my_outputs = self(input_ids.permute(1,0)).permute(1,0,2)
        outputs_mean = torch.mean(my_outputs - BERT_outputs)
        outputs_norm = torch.norm(my_outputs - BERT_outputs)
        
        print(f"Integrity test returns mean {outputs_mean} and norm {outputs_norm}")
        self.train()
        

class BERTStyleDecoder(nn.Module):
    def __init__(self, vocab_size=30522, dim_model=768, nhead=12, name="music_bert",\
                 num_encoder_layers=12, d_feed=3072, dropout=0.1):
        super(BERTStyleDecoder, self).__init__()
        self.vocab_size = vocab_size
        
        self.embedder = BERTStyleEmbedding(dim_model, self.vocab_size)
        
        encoder_layer = TransformerDecoderLayer(dim_model, nhead, \
                                                dim_feedforward=d_feed, \
                                                dropout=dropout,
                                                activation='gelu')
        
        self.decoder = TransformerDecoder(encoder_layer, num_encoder_layers)
    
    def forward(self, src_tokens):
        
        sequence = self.embedder(src_tokens)
        output = self.decoder(sequence)
        
        return output
    
    def get_device(self):
        return next(self.parameters()).device
    
    def load_pretrained(self):
        #load pretrained model from Hugginface
        
        device = self.get_device()
        bertModel = BertModel.from_pretrained("bert-base-uncased").to(device)
        
        script_dir = os.path.dirname(__file__)
        with open(os.path.join(script_dir, "layer_maps/bert_decoder_extras_map.json"), "r") as f:
            extras_map = json.load(f)
        with open(os.path.join(script_dir, "layer_maps/bert_decoder_layer_map.json"), "r") as f:
            layer_map = json.load(f)
        with open(os.path.join(script_dir, "layer_maps/bert_decoder_self_attention_map.json"), "r") as f:
            self_att_map = json.load(f)
        
        bert_state_dict = translate_from_hugginface_to_torch_BERT(bertModel.state_dict(), extras_map, layer_map, self_att_map)
        unset_keys = self.load_state_dict(bert_state_dict, strict=False)
        
    
        return unset_keys