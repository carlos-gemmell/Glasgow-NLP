import torch
import torchvision
from torch.nn.modules.module import Module
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.linear import Linear
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.core.lightning import LightningModule
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm
from torch.nn.modules.container import ModuleList

import copy

class Cluster_Transformer(LightningModule):
    def __init__(self, embed_dim=768, vocab_size=32000, num_layers=4, **kwargs):
        super().__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        encoder_layer = TransformerEncoderLayer(d_model=embed_dim)
        self.encoder = TransformerEncoder(encoder_layer, num_layers)
        
        decoder_layer = TransformerDecoderLayer(d_model=embed_dim)
        self.decoder = TransformerDecoder(encoder_layer, num_layers)
        
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        
        self.proj_layer = nn.Linear(embed_dim, vocab_size)
        
    def transformer_encoder(self, src, src_key_padding_mask=None):
        output = src
        for i in range(self.num_layers):
            output = self.encoder_layers[i](output, src_key_padding_mask=src_key_padding_mask)
        return output
    
    def forward(self, src, src_key_padding_mask=None):
        '''
        S sequence length, N Batch size, E embeding dim
        input_ids: [S,N] 
        '''
        src = self.embedding(input_ids)
        src = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        outputs = self.proj_layer(src)
        return outputs
    
    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        input_mask = batch['input_attention_mask']
        
        target_ids = batch['target_ids']
                
        logits = self(input_ids, src_key_padding_mask=input_mask)  
                
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.vocab_size), target_ids.view(-1)) 
        return {"loss":loss, 'logits':logits}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
        return [optimizer], [scheduler]
    
    def backward(self, use_amp, loss, optimizer, _):
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        

class Transformer(Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", custom_encoder=None, custom_decoder=None):
        super(Transformer, self).__init__()

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
            encoder_norm = None # LayerNorm(d_model)
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
            decoder_norm = LayerNorm(d_model)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def forward(self, src, tgt, src_mask=None, tgt_mask=None,
                memory_mask=None, src_key_padding_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):

        if src.size(1) != tgt.size(1):
            raise RuntimeError("the batch number of src and tgt must be equal")

        if src.size(2) != self.d_model or tgt.size(2) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output, atts = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        return output, atts


    def generate_square_subsequent_mask(self, sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)



class TransformerEncoder(Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):

        for i in range(self.num_layers):
            outputs = self.layers[i](src, src_mask=mask,
                                    src_key_padding_mask=src_key_padding_mask)
            src, queries, keys, slot_vectors, slot_assignment_scores, slot_attention_scores = outputs

        if self.norm:
            src = self.norm(src)

        return src, queries, keys, slot_vectors, slot_assignment_scores, slot_attention_scores



class TransformerDecoder(Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask=None,
                memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        r"""Pass the inputs (and mask) through the decoder layer in turn.
        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequnce from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
        """

        for i in range(self.num_layers):
            output = self.layers[i](tgt, memory, tgt_mask=tgt_mask,
                                    memory_mask=memory_mask,
                                    tgt_key_padding_mask=tgt_key_padding_mask,
                                    memory_key_padding_mask=memory_key_padding_mask)
            tgt, queries, keys, slot_vectors, slot_assignment_scores, slot_attention_scores = outputs

        if self.norm:
            tgt = self.norm(tgt)

        return tgt, queries, keys, slot_vectors, slot_assignment_scores, slot_attention_scores



class TransformerDecoderLayer(Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.slot_attn = Hierarchical_Attention(d_model, cycles=1)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

    def forward(self, tgt, memory, slot_vectors, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequnce from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
        """
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        outputs = self.slot_attn(tgt, memory, memory, slot_vectors=slot_vectors,
                                   key_padding_mask=memory_key_padding_mask)
        tgt2, queries, keys, slot_vectors, slot_assignment_scores, slot_attention_scores = outputs
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(F.gelu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, queries, keys, slot_vectors, slot_assignment_scores, slot_attention_scores


class TransformerEncoderLayer(Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1, activation="relu"):
        '''
        encoder_layer = TransformerEncoderLayer(d_model=512, nhead=8)
        src = torch.rand(10, 32, 512)
        encoder_layer(src).shape
        '''
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = Hierarchical_Attention(d_model, cycles=3)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        r"""Pass the input through the endocder layer.
        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        """
        outputs = self.self_attn(src, src, src, key_padding_mask=src_key_padding_mask)
        scaled_values, queries, keys, slot_vectors, slot_assignment_scores, slot_attention_scores = outputs
        src = src + self.dropout1(scaled_values)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.gelu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, queries, keys, slot_vectors, slot_assignment_scores, slot_attention_scores


class Hierarchical_Attention(Module):
    def __init__(self, embed_dim, dropout=0.1, dim_feedforward=128, cycles=3, passthrough_mode=False, q_k_sim='dot'):
        '''
        Hierarchical attention is a way to put keys in long sequences into slots/buckets 
        to improve sparcity and time-space efficiency from O(n^2) to O(n log n)
        This is achieved in two passes, first we populates the slots with representative 
        samples of the tokens bellow. Then when computing token level attention, the queries
        are compared to the slots first, then the derived attention scores weigh the tokens 
        and lower level attention scores under under that slot. 
        '''
        super().__init__()
        self.embed_dim = embed_dim
        self.cycles = cycles
        self.passthrough_mode = passthrough_mode
        self.q_k_sim = q_k_sim
        
        self.scaling = float(embed_dim) ** -0.5
        
        self.slot_Wq = Linear(embed_dim,embed_dim, bias=False)
        self.slot_Wk = Linear(embed_dim,embed_dim, bias=False)
        self.slot_Wv = Linear(embed_dim,embed_dim, bias=False)
        
        self.Wq = Linear(embed_dim,embed_dim, bias=False)
        self.Wk = Linear(embed_dim,embed_dim, bias=False)
        self.Wv = Linear(embed_dim,embed_dim, bias=False)
                
        self.linear1 = Linear(embed_dim, dim_feedforward)
        self.linear2 = Linear(dim_feedforward, embed_dim)
        
        if passthrough_mode:
            dropout = 0
            self.slot_Wq.weight.data = torch.eye(embed_dim,embed_dim)
            self.slot_Wk.weight.data = torch.eye(embed_dim,embed_dim)
            self.slot_Wv.weight.data = torch.eye(embed_dim,embed_dim)
            
            self.Wq.weight.data = torch.eye(embed_dim,embed_dim)
            self.Wk.weight.data = torch.eye(embed_dim,embed_dim)
            self.Wv.weight.data = torch.eye(embed_dim,embed_dim)
            
            self.linear1.weight.data = torch.eye(dim_feedforward, embed_dim)
            self.linear2.weight.data = torch.eye(embed_dim, dim_feedforward)
            self.linear1.bias.data = torch.zeros((dim_feedforward,))
            self.linear2.bias.data = torch.zeros((embed_dim,))
            
            self.norm1 = lambda x: x
            self.norm2 = lambda x: x
            
            self.scaling = 1.0
        else: 
            self.norm1 = LayerNorm(embed_dim)
            self.norm2 = LayerNorm(embed_dim)

        self.dropout = Dropout(dropout)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        
    def slot_transformer_layer(self, token_vects, slot_vects):     
        Q = self.slot_Wq(slot_vects) # [N,L,E]
        K = self.slot_Wk(token_vects) # [N,L,E]
        V = self.slot_Wv(token_vects) # [N,L,E]
        
#         print('in transformer token_vects', token_vects)
#         print('in transformer slot_vects', slot_vects)
#         print('in transformer Q', Q)
#         print('in transformer K', K)
#         print('in transformer V', V)
        
        if self.q_k_sim == 'dot':
            Q = Q*self.scaling
            attention_matrix = torch.bmm(Q,K.permute(0,2,1))
        if self.q_k_sim == 'euclid':
            attention_matrix = torch.cdist(Q, K, p=2.0)
#         print('in transformer attention_matrix', attention_matrix)
        attention_scores = torch.softmax(attention_matrix, dim=-1)
#         print('in transformer attention_scores', attention_scores)
        slot_assignment_scores = torch.softmax(attention_matrix, dim=-2)
#         print('in transformer slot_assignment_scores', slot_assignment_scores)
        src2 = torch.bmm(attention_scores, V)
#         print('in transformer src2', src2)
        src = slot_vects
        src = src + self.dropout(src2)
        
        if self.passthrough_mode:
            return src, slot_assignment_scores
        
        src = self.norm1(src)
        src2 = self.linear2(self.dropout1(F.gelu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, slot_assignment_scores
        
    def forward(self, queries, keys, values, slot_vectors=None , cutof='soft', key_padding_mask=None):
        '''
        L sequence length, S sequence length, N Batch size, E embeding dim
        queries: [L,N,E]
        keys: [S,N,E]
        values: [S,N,E]
        slot_vectors: [S,N,E]
        key_padding_mask: [N, S]: padding tensor, 0's for real tokens, 1's for padding tokens
        
        returns: [L,N,E]
        >>> token_vects = torch.eye(4,4).unsqueeze(1)
        >>> h_att = Hierarchical_Attention(4, cycles=1, passthrough_mode=True)
        >>> h_att(token_vects,  token_vects, token_vects)
        tensor([[[0.3854, 0.2049, 0.2049, 0.2049]],
                [[0.2049, 0.3854, 0.2049, 0.2049]],
                [[0.2151, 0.2151, 0.3547, 0.2151]],
                [[0.2151, 0.2151, 0.2151, 0.3547]]], grad_fn=<PermuteBackward>)
        '''
        seq_len = queries.shape[0]
        batch_size = queries.shape[1]
        num_slots = int(torch.floor(torch.log2(torch.tensor(seq_len, dtype=torch.float))))
        
        if self.passthrough_mode:
            slot_vectors = torch.eye(num_slots,self.embed_dim).unsqueeze(0).repeat(batch_size,1,1)
        elif torch.is_tensor(slot_vectors):
            slot_vectors = slot_vectors.permute(1,0,2)
        else:
            slot_vectors = torch.rand((batch_size, num_slots, self.embed_dim))
        
        queries = queries.permute(1,0,2)
        keys = keys.permute(1,0,2)
        values = values.permute(1,0,2)
        
#         print("init slot_vectors", slot_vectors)
        
        # compute slot vectors
        for i in range(self.cycles):
            slot_vectors, slot_assignment_scores = self.slot_transformer_layer(queries, slot_vectors)
            # slot_vectors [N, n_buckets, emb_dim]  slot_assignment_scores [N, n_buckets, L]
#         print("final slot_vectors", slot_vectors)
#         print("final slot_assignment_scores", slot_assignment_scores)
            
        hard_slot_assignment_scores = torch.argmax(slot_assignment_scores, dim=-2)
#         print("final hard_slot_assignment_scores", hard_slot_assignment_scores)
        
        # compute slot attention scores
        Q = self.Wq(queries) # [N,L,E]
        K = self.Wk(keys) # [N,L,E]
        V = self.Wv(values) # [N,L,E]
        
#         print("token Q", Q)
#         print("token K", K)
#         print("token V", V)
        if self.q_k_sim == 'dot':
            Q = Q*self.scaling
            slot_attention_matrix = torch.bmm(Q,slot_vectors.permute(0,2,1))
        if self.q_k_sim == 'euclid':
            slot_attention_matrix = torch.cdist(Q, slot_vectors, p=2.0)
#         print("slot_attention_matrix", slot_attention_matrix)
        slot_attention_scores = torch.softmax(slot_attention_matrix, dim=-1)
#         print("slot_attention_scores", slot_attention_scores)
        
        sorted_slot_attention_scores = torch.argsort(slot_attention_scores, dim=-1)
#         print("sorted_slot_attention_scores", sorted_slot_attention_scores)
        
        token_assignment_att_scores = torch.bmm(slot_attention_scores, slot_assignment_scores) # this needs to be reviewed, I think it's wrong
#         print("token_assignment_att_scores", token_assignment_att_scores)
        
        token_attention_matrix = torch.bmm(Q,K.permute(0,2,1))
#         print("token_attention_matrix", token_attention_matrix)
        scaled_token_attention_matrix = torch.mul(token_attention_matrix, token_assignment_att_scores)
#         print("scaled_token_attention_matrix", scaled_token_attention_matrix)
        
        if torch.is_tensor(key_padding_mask):
            scaled_token_attention_matrix = scaled_token_attention_matrix.masked_fill(
                key_padding_mask==1.0,
                float('-inf'),
            )
#         print('masked scaled_token_attention_matrix',scaled_token_attention_matrix)
        
        scaled_token_attention_matrix_scores = torch.softmax(scaled_token_attention_matrix, dim=-1)
#         print("scaled_token_attention_matrix_scores", scaled_token_attention_matrix_scores)
        
        scaled_values = torch.bmm(scaled_token_attention_matrix_scores, V)
#         print("scaled_values", scaled_values)
        
        scaled_values = scaled_values.permute(1,0,2)
        output = (scaled_values, Q.permute(1,0,2), K.permute(1,0,2), slot_vectors.permute(1,0,2), slot_assignment_scores, slot_attention_scores)
        return output
    
    
    
    
def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])