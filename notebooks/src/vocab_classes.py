import collections
from transformers import BertTokenizer

class Shared_Vocab():
    def __init__(self, data, vocab_size, tokenizer_fn, use_OOVs=False):
        
        self.stoi = {"<unk>":0, "<sos>":1, "<eos>":2, "<pad>":3}
        self.vocab_size = vocab_size  - len(self.stoi)
        self.tokenizer_fn = tokenizer_fn
        self.use_OOVs = use_OOVs
        
        self.SOS = self.stoi["<sos>"]
        self.EOS = self.stoi["<eos>"]
        self.PAD = self.stoi["<pad>"]
        self.UNK = self.stoi["<unk>"]
        
        self.OOV_stoi = {}
        self.OOV_itos = {}
        self.OOV_starter_count = 30000
        self.OOV_count = self.OOV_starter_count
        
        all_toks = []
        for (src, tgt) in data:
            all_toks += tokenizer_fn(src)
            all_toks += tokenizer_fn(tgt)

        most_freq = collections.Counter(all_toks).most_common(self.vocab_size)

        for tok, count in most_freq:
            self.stoi[tok] = len(self.stoi)
        
        self.itos = [k for k,v in sorted(self.stoi.items(), key=lambda kv: kv[1])]
        self.size = len(self.itos)
        
        
    def encode_input(self, string):
        OOVs = []
        IDs = []
        words = self.tokenizer_fn(string)
        for word in words:
            try:
                id = self.stoi[word]
                IDs.append(id)
            except KeyError as e:
                # word is OOV
                if self.use_OOVs:
                    IDs.append(len(self.stoi) + len(OOVs))
                    OOVs.append(word)
                else:
                    IDs.append(self.stoi["<unk>"])
        
        if self.use_OOVs: 
            OOV_ids = []

            for OOV in OOVs:
                try:
                    idx = self.OOV_stoi[OOV]
                    OOV_ids.append(idx)
                except KeyError as e:
                    self.OOV_count += 1
                    self.OOV_stoi[OOV] = self.OOV_count
                    self.OOV_itos[self.OOV_count] = OOV
                    OOV_ids.append(self.OOV_count)
            return IDs, OOV_ids
        else: return IDs
    
    def encode_output(self, string, OOV_ids=[]):
        OOVs = self.OOV_ids2OOVs(OOV_ids)
        IDs = []
        words = self.tokenizer_fn(string)
        for word in words:
            try:
                id = self.stoi[word]
                IDs.append(id)
            except KeyError as e:
                # word is OOV
                try:
                    IDs.append(len(self.stoi) + OOVs.index(word))
                except ValueError as e:
                    IDs.append(self.stoi["<unk>"])
        return IDs
    
    def decode(self, ids, OOV_ids=[], copy_marker="(COPY)"):
        OOVs = self.OOV_ids2OOVs(OOV_ids)
        extended_itos = self.itos.copy()
        extended_itos += [OOV+copy_marker for OOV in OOVs]
        return " ".join([extended_itos[id] for id in ids if id<len(extended_itos)])
    
    def decode_input(self, ids, OOVs=[], copy_marker="(COPY)"):
        return self.decode(ids, OOVs,copy_marker=copy_marker)
    
    def decode_output(self, ids, OOVs=[], copy_marker="(COPY)"):
        return self.decode(ids, OOVs,copy_marker=copy_marker)
    
    def OOV_ids2OOVs(self, OOV_ids):
        return [self.OOV_itos[OOV_id] if OOV_id in self.OOV_itos else "<unk>" for OOV_id in OOV_ids]
    
    def add_test_set_OOVs(self, data):
        """
        data: [(src,tgt)]
        """
        for (src, tgt) in data:
            self.encode_input(src)
    
    
class BERT_Vocab():
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        special_tokens_dict = {'bos_token': '[unused0]','eos_token': '[unused1]'}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.SOS = self.tokenizer.bos_token_id
        self.EOS = self.tokenizer.eos_token_id
        self.PAD = self.tokenizer.pad_token_id
        self.UNK = self.tokenizer.unk_token_id
        
        self.vocab_size = self.tokenizer.vocab_size
        
    def encode_output(self, string, OOV_ids=[]):
        return self.tokenizer.encode(string)[1:-1]
    def encode_input(self, string):
        return self.tokenizer.encode(string)[1:-1], []
    def decode_input(self, ids, OOVs=[]):
        return " ".join(self.tokenizer.convert_ids_to_tokens(ids))
    def decode_output(self, ids, OOVs=[]):
        return " ".join(self.tokenizer.convert_ids_to_tokens(ids))