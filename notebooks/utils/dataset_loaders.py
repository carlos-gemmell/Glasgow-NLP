from .file_ops import corpus_to_array
from .useful_utils import filter_corpus, clean_samples, string_split_v3

class SRC_TGT_pairs():
    def __init__(self, src_fp, tgt_fp, max_seq_len=50, split_fn=string_split_v3):
        self.samples = corpus_to_array(src_fp, tgt_fp)
        self.samples = filter_corpus(self.samples, max_seq_len, tokenizer=split_fn)
        self.samples = clean_samples(self.samples)
        
    def get_samples(self):
        return self.samples
        