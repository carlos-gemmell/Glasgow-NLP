from .file_ops import corpus_to_array
from .useful_utils import filter_corpus, clean_samples, string_split_v3

from torchtext.data import Field, BucketIterator
import torchtext
from dotmap import DotMap

class SRC_TGT_pairs():
    def __init__(self, src_fp, tgt_fp, max_seq_len=50, split_fn=string_split_v3):
        self.samples = corpus_to_array(src_fp, tgt_fp)
        self.samples = filter_corpus(self.samples, max_seq_len, tokenizer=split_fn)
        self.samples = clean_samples(self.samples)
        
    def get_samples(self):
        return self.samples
        
        
        
def data2dataset_OOV(data, vocab):
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
            src_ids, OOVs = vocab.encode_input(src)
            tgt_ids = vocab.encode_output(tgt, OOVs)
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