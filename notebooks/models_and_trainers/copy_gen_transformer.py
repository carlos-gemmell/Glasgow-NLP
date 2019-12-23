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

from IPython.core.debugger import set_trace as tr
from .base_transformer import TransformerModel, PositionalEncoding
from dotmap import DotMap

class CopyGeneratorModel():
    def __init__(self, vocab, vocab_size, embed_dim, att_heads, layers, dim_feedforward, lr, max_seq_length, use_copy=True):
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.model = CopyGeneratorTransformer(vocab_size, 
                                             embed_dim, 
                                             att_heads, 
                                             layers, 
                                             dim_feedforward, 
                                             dropout=0.5, 
                                             use_copy=True)
        self.max_seq_length = max_seq_length
        self.use_copy = use_copy
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.99)
        
        
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
    
    def data2dataset_OOV(self, data):
        TEXT_FIELD = Field(sequential=True, use_vocab=False, unk_token=0, init_token=1,eos_token=2, pad_token=3)
        OOV_TEXT_FIELD = Field(sequential=True, use_vocab=False, pad_token=3)

        OOV_stoi = {}
        OOV_itos = {}
        OOV_starter_count = 30000
        OOV_count = OOV_starter_count
        metadata = DotMap()
        metadata.OOV_stoi = OOV_stoi
        metadata.OOV_itos = OOV_itos

        examples = []

        for (src, tgt) in data:
            src_ids, OOVs = self.vocab.encode_input(src)
            tgt_ids = self.vocab.encode_output(tgt, OOVs)
            OOV_ids = []

            for OOV in OOVs:
                try:
                    idx = OOV_stoi[OOV]
                    OOV_ids.append(idx)
                except KeyError as e:
                    OOV_count += 1
                    OOV_stoi[OOV] = OOV_count
                    OOV_itos[OOV_count] = OOV
                    OOV_ids.append(OOV_count)

            examples.append(torchtext.data.Example.fromdict({"src":src_ids, 
                                                             "tgt":tgt_ids, 
                                                             "OOVs":OOV_ids}, 
                                                    fields={"src":("src",TEXT_FIELD), 
                                                            "tgt":("tgt",TEXT_FIELD), 
                                                            "OOVs":("OOVs", OOV_TEXT_FIELD)}))
        
        dataset = torchtext.data.Dataset(examples,fields={"src":TEXT_FIELD, 
                                                          "tgt":TEXT_FIELD, 
                                                          "OOVs":OOV_TEXT_FIELD})
        
        return dataset, metadata
    
    def batch_filter_ids(self, batch_list):
        return [[id for id in l if id not in [1,2,3]] for l in batch_list]
    
    def evaluate_iterator(self, iterator, metadta):
        BLEU_scores = []
        for i, batch in enumerate(valid_iterator):
            batch_size = batch.src.shape[1]
            
            encoder_inputs = batch.src
            predictions = beam_search.beam_search_decode(model,
                              batch_encoder_ids=encoder_inputs,
                              SOS_token=stoi["<sos>"],
                              EOS_token=stoi["<eos>"],
                              PAD_token=stoi["<pad>"],
                              beam_size=beam_size,
                              max_length=30,
                              num_out=1)
            
            sources = encoder_inputs.transpose(0,1).cpu().tolist()
            sources = batch_filter_ids(sources)
            
            predictions = [t[0].view(-1).cpu().tolist() for t in predictions]
            predictions = self.batch_filter_ids(predictions)
            
            targets = batch.tgt.transpose(0,1).cpu().tolist()
            targets = batch_filter_ids(targets)
            
#             print(batch.tgt)
            
            OOVss = [[OOV_itos[OOV] for OOV in batch.OOVs.cpu()[:,idx].tolist() if OOV != 3] for idx in range(batch_size)]
            
            if i % int(len(valid_iterator)/3) == 0:
                print("| EVALUATION | {:5d}/{:5d} batches |".format(i, len(valid_iterator)))
            
            for j in range(batch_size):
                BLEU = nltk_bleu(targets[j], predictions[j])
                BLEU_scores.append(BLEU)
                
                out_fp.write("SRC  :" + decode(sources[j], OOVss[j]) + "\n")
                out_fp.write("TGT  :" + decode(targets[j], OOVss[j]) + "\n")
                out_fp.write("PRED :" + decode(predictions[j], OOVss[j]) + "\n")
                out_fp.write("BLEU :" + str(BLEU) + "\n")
                out_fp.write("\n")
                
                if log:
                    print("SRC  :" + decode(sources[j], OOVss[j]))
                    print("TGT  :" + decode(targets[j], OOVss[j]))
                    print("PRED :" + decode(predictions[j], OOVss[j]))
                    print("BLEU :" + str(BLEU))
                    print()
        out_fp.write("\n\n| EVALUATION | BLEU: {:5.2f} |\n".format(np.average(BLEU_scores)))
        print("| EVALUATION | BLEU: {:5.3f} |".format(np.average(BLEU_scores)))
        return np.average(BLEU_scores)
        
        
    def save(save_fp):
        torch.save((self.model, self.optimizer, self.scheduler, self.vocab), save_fp)
    
    def restore(restore_fp):
        self.model, self.optimizer, self.scheduler, self.vocab = torch.load(restore_fp)


class CopyGeneratorTransformer(nn.Module):

    def __init__(self, vocab_size, embed_dim, att_heads, layers, dim_feedforward, dropout=0.5, use_copy=True):
        super(CopyGeneratorTransformer, self).__init__()
        self.model_type = 'Transformer'
        
        self.use_copy = use_copy
        self.embedding_size = embed_dim
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        self.src_encoder = nn.Embedding(vocab_size, embed_dim)
        self.tgt_encoder = nn.Embedding(vocab_size, embed_dim)
        
        self.transformer = Transformer(d_model=embed_dim,
                                       nhead=att_heads, 
                                       num_encoder_layers=layers, 
                                       num_decoder_layers=layers, 
                                       dim_feedforward=dim_feedforward)
        self.decoder = nn.Linear(embed_dim, vocab_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.att_mask_noise = 0.0
        
        if use_copy:
            self.p_generator = nn.Linear(embed_dim,1)

        self.init_weights()
        self.tgt_mask = None

    def init_weights(self):
        initrange = 0.1
        self.src_encoder.weight.data.uniform_(-initrange, initrange)
        self.tgt_encoder.weight.data.uniform_(-initrange, initrange)
        
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
        self.tgt_mask = self._generate_square_subsequent_mask(len(tgt)).to(self.device)
        

        src_emb = self.src_encoder(src) * math.sqrt(self.embedding_size)
        src_emb = self.pos_encoder(src_emb)
        
        tgt_emb = self.tgt_encoder(tgt) * math.sqrt(self.embedding_size)
        tgt_emb = self.pos_encoder(tgt_emb)
        
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

class Transformer(Module):
    r"""A transformer model. User is able to modify the attributes as needed. The architecture
    is based on the paper "Attention Is All You Need". Ashish Vaswani, Noam Shazeer,
    Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and
    Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information
    Processing Systems, pages 6000-6010. Users can build the BERT(https://arxiv.org/abs/1810.04805)
    model with corresponding parameters.

    Args:
        d_model: the number of expected features in the encoder/decoder inputs (default=512).
        nhead: the number of heads in the multiheadattention models (default=8).
        num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
        num_decoder_layers: the number of sub-decoder-layers in the decoder (default=6).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of encoder/decoder intermediate layer, relu or gelu (default=relu).
        custom_encoder: custom encoder (default=None).
        custom_decoder: custom decoder (default=None).

    Examples::
        >>> transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
        >>> src = torch.rand((10, 32, 512))
        >>> tgt = torch.rand((20, 32, 512))
        >>> out = transformer_model(src, tgt)

    Note: A full example to apply nn.Transformer module for the word language model is available in
    https://github.com/pytorch/examples/tree/master/word_language_model
    """

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", custom_encoder=None, custom_decoder=None):
        super(Transformer, self).__init__()

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
            encoder_norm = LayerNorm(d_model)
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
        r"""Take in and process masked source/target sequences.

        Args:
            src: the sequence to the encoder (required).
            tgt: the sequence to the decoder (required).
            src_mask: the additive mask for the src sequence (optional).
            tgt_mask: the additive mask for the tgt sequence (optional).
            memory_mask: the additive mask for the encoder output (optional).
            src_key_padding_mask: the ByteTensor mask for src keys per batch (optional).
            tgt_key_padding_mask: the ByteTensor mask for tgt keys per batch (optional).
            memory_key_padding_mask: the ByteTensor mask for memory keys per batch (optional).

        Shape:
            - src: :math:`(S, N, E)`.
            - tgt: :math:`(T, N, E)`.
            - src_mask: :math:`(S, S)`.
            - tgt_mask: :math:`(T, T)`.
            - memory_mask: :math:`(T, S)`.
            - src_key_padding_mask: :math:`(N, S)`.
            - tgt_key_padding_mask: :math:`(N, T)`.
            - memory_key_padding_mask: :math:`(N, S)`.

            Note: [src/tgt/memory]_mask should be filled with
            float('-inf') for the masked positions and float(0.0) else. These masks
            ensure that predictions for position i depend only on the unmasked positions
            j and are applied identically for each sequence in a batch.
            [src/tgt/memory]_key_padding_mask should be a ByteTensor where True values are positions
            that should be masked with float('-inf') and False values will be unchanged.
            This mask ensures that no information will be taken from position i if
            it is masked, and has a separate mask for each sequence in a batch.

            - output: :math:`(T, N, E)`.

            Note: Due to the multi-head attention architecture in the transformer model,
            the output sequence length of a transformer is same as the input sequence
            (i.e. target) length of the decode.

            where S is the source sequence length, T is the target sequence length, N is the
            batch size, E is the feature number

        Examples:
            >>> output = transformer_model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        """

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
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        r"""Pass the input through the endocder layers in turn.

        Args:
            src: the sequnce to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for i in range(self.num_layers):
            output = self.layers[i](output, src_mask=mask,
                                    src_key_padding_mask=src_key_padding_mask)

        if self.norm:
            output = self.norm(output)

        return output



class TransformerDecoder(Module):
    r"""TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = transformer_decoder(tgt, memory)
    """

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

        Shape:
            see the docs in Transformer class.
        """
        output = tgt

        for i in range(self.num_layers):
            output, atts = self.layers[i](output, memory, tgt_mask=tgt_mask,
                                    memory_mask=memory_mask,
                                    tgt_key_padding_mask=tgt_key_padding_mask,
                                    memory_key_padding_mask=memory_key_padding_mask)

        if self.norm:
            output = self.norm(output)

        return output, atts # the attention weights are from the last layer only


class TransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        r"""Pass the input through the endocder layer.

        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        if hasattr(self, "activation"):
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        else:  # for backward compatibility
            src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src



class TransformerDecoderLayer(Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
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

        self.activation = _get_activation_fn(activation)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequnce from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, atts = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        if hasattr(self, "activation"):
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        else:  # for backward compatibility
            tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, atts



def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise RuntimeError("activation should be relu/gelu, not %s." % activation)