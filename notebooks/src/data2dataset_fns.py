from .useful_utils import string_split_v3, string_split_v2, string_split_v1
from .metrics import nltk_bleu
from .vocab_classes import Shared_Vocab
from transformers import BertTokenizer

import torchtext
from torchtext.data import Field, BucketIterator


def data2dataset_shared_vocab_with_OOVs(data, encode_to_example_fn, vocab):
    TEXT_FIELD = Field(sequential=True, use_vocab=False, unk_token=vocab.UNK, init_token=vocab.SOS,eos_token=vocab.EOS, pad_token=vocab.PAD)
    OOV_TEXT_FIELD = Field(sequential=True, use_vocab=False, pad_token=vocab.PAD)
    
    fields = {"src":TEXT_FIELD, "tgt":TEXT_FIELD, "OOVs":OOV_TEXT_FIELD}

    examples = []

    for (src, tgt) in data:
        example = encode_to_example_fn(src, tgt, fields)

        examples.append(example)
    dataset = torchtext.data.Dataset(examples,fields=fields)
    return dataset

def BERTEncode(vocab):
    def encode_to_example(src, tgt, fields):
        
        
        src_ids = vocab.encode_input(src)
        tgt_ids = vocab.encode_input(tgt)
#         src_ids = [vocab.SOS] + src_ids + [vocab.EOS]
#         tgt_ids = [vocab.SOS] + tgt_ids + [vocab.EOS]
        
        example = torchtext.data.Example.fromdict({"src":src_ids, 
                                                   "tgt":tgt_ids, 
                                                   "OOVs":[]}, 
                                                        fields={"src":("src",fields["src"]), 
                                                                "tgt":("tgt",fields["tgt"]), 
                                                                "OOVs":("OOVs", fields["OOVs"])})
        return example
    
    return encode_to_example


def CustomEncode(vocab):
    def encode_to_example(src, tgt, fields):
        
        src_ids, OOV_ids = vocab.encode_input(src)
        tgt_ids = vocab.encode_output(tgt, OOV_ids)
        
        example = torchtext.data.Example.fromdict({"src":src_ids, 
                                                   "tgt":tgt_ids, 
                                                   "OOVs":OOV_ids}, 
                                                        fields={"src":("src",fields["src"]), 
                                                                "tgt":("tgt",fields["tgt"]), 
                                                                "OOVs":("OOVs", fields["OOVs"])})
        return example
    
    return encode_to_example