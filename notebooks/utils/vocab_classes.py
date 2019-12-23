import collections

class Shared_Vocab():
    def __init__(self, data, vocab_size, tokenizer_fn, use_OOVs=False):
        
        self.stoi = {"<unk>":0, "<sos>":1, "<eos>":2, "<pad>":3}
        self.vocab_size = vocab_size  - len(self.stoi)
        self.tokenizer_fn = tokenizer_fn
        self.use_OOVs = use_OOVs
        
        all_toks = []
        for (src, tgt) in data:
            all_toks += tokenizer_fn(src)
            all_toks += tokenizer_fn(tgt)

        most_freq = collections.Counter(all_toks).most_common(self.vocab_size)

        for tok, count in most_freq:
            self.stoi[tok] = len(self.stoi)
        
        self.itos = [k for k,v in sorted(self.stoi.items(), key=lambda kv: kv[1])]
        
        
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
        
        if self.use_OOVs: return IDs, OOVs
        else: return IDs
    
    def encode_output(self, string, OOVs=[]):
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
    
    def decode(self, ids, OOVs=[]):
        extended_itos = self.itos.copy()
        extended_itos += [OOV+"(COPY)" for OOV in OOVs]
        return " ".join([extended_itos[id] for id in ids if id<len(extended_itos)])
    
    def decode_input(self, ids, OOVs=[]):
        return decode(ids, OOVs=[])
    
    def decode_output(self, ids, OOVs=[]):
        return decode(ids, OOVs=[])
    
    
class Separate_Vocab():
    def __init__(self, data, vocab_size, src_tokenizer_fn, tgt_tokenizer_fn):
        pass