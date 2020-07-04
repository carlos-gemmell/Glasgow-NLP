import os
from abc import ABC, abstractmethod
import torch
import json
import numpy as np
from tokenizers import ByteLevelBPETokenizer
from torch.utils.data import Dataset, DataLoader
from src.tree_sitter_AST_utils import Tree_Sitter_ENFA, sub_str_from_coords, Node_Processor, \
                                        Code_Parser, StringTSNode, get_grammar_vocab, regex_to_member, \
                                        NodeBuilder, PartialNode, sub_str_from_coords, PartialTree
import tqdm
def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')
if is_interactive():
    import tqdm.notebook as tqdm 

class DataProcessor(ABC):
    @abstractmethod
    def __init__(self, task_data):
        """
        Things like max sequence length should be passed here and enforced elsewhere.
        If the dataset is too large to fit, this is where the transformations will happen
        to save the samples in a database or use IDs to make transfer faster.
        """
        self.task_data = task_data
    
#     @abstractmethod
    def encode(self, input_samples):
        """
        This function needs to produce all the necessary outputs for a model 
        to only take this as input and produce a correct output.
        """
        pass
    
#     @abstractmethod
    def decode(self, output_tensor):
        """
        This funciton should produce a correct output for the specified task, 
        it doesn't need to be the same for every transformation or task.
        """
        pass
    
    @abstractmethod
    def to_dataloader(self, batch_size, repeat=False):
        pass
    
    @abstractmethod
    def save(self, path):
        """
        This should save the entire object for easy access. Saving and Loading is
        specific to the dataProcessors.
        """
        pass
        
    def load(path):
        """
        STATIC METHOD
        will load all kinds of data processors using torch.
        """
        return torch.load(path)
    
class CodeTrainedBPE_Translation_DataProcessor(DataProcessor, Dataset):
    def __init__(self, task_data, max_src_len=512, max_tgt_len=512):
        """
        This data processor tokenizes and numericalises using a custom byte pair 
        encoding trained on the codeSearchNet train data with full docstrings.
        """
        self.task_data = task_data
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.tokenizer = ByteLevelBPETokenizer("/nfs/phd_by_carlos/notebooks/datasets/code_search_net/code_bpe_hugging_32k-vocab.json",
                                          "/nfs/phd_by_carlos/notebooks/datasets/code_search_net/code_bpe_hugging_32k-merges.txt")
        self.tokenizer.add_special_tokens(["[CLS]", "[SOS]", "[EOS]", "[PAD]"])
        self.SOS = self.tokenizer.encode("[SOS]").ids[0]
        self.EOS = self.tokenizer.encode("[EOS]").ids[0]
        self.PAD = self.tokenizer.encode("[PAD]").ids[0]
        self.CLS = self.tokenizer.encode("[CLS]").ids[0]
        
        self.__remove_long_samples()
        
        
    def __len__(self):
        return len(self.task_data)
    
    def __getitem__(self, idx):
        src, tgt = self.task_data[idx]
        sample = {'src': self.encode(src), 'tgt': self.encode(tgt)}
        return sample
    
    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()
    
    def __remove_long_samples(self):
        for i in tqdm.tqdm(list(reversed(range(len(self.task_data)))), desc="removing long samples"):
            src, tgt = self.task_data[i]
            if len(self.encode(src))>self.max_src_len or len(self.encode(tgt))>self.max_tgt_len:
                del self.task_data[i]
        
    def encode(self, sample):
        """
        sample: str: the input string to encode
        """
        return [self.SOS] + self.tokenizer.encode(sample).ids + [self.EOS]
    
    def encode_src(self, sample):
        return self.encode(sample)
    def encode_tgt(self, sample):
        return self.encode(sample)
        
    def encode_to_tensor(self,input_samples):
        """
        input_samples: [str]: one or more strings to convert to a single padded tensor. (Seq_len x batch)
        """
        return pad_sequence([torch.Tensor(self.encode(sample)).type(torch.LongTensor) for sample in input_samples], padding_value=self.PAD)
    
    def collate(self, input_samples):
        """
        input_samples: [dict]: these are samples obtained through the _get_item method
        """
        collated_samples = {}
        sample_keys = input_samples[0].keys()
        for key in sample_keys:
            collated_samples[key] = torch.nn.utils.rnn.pad_sequence([torch.Tensor(sample[key]).type(torch.LongTensor) for sample in input_samples], 
                                                 padding_value=self.PAD)
        return collated_samples
        
    def decode(self, ids):
        """
        ids: [int]: ids to decode
        """
        return self.tokenizer.decode(ids)
    
    def decode_src(self, ids):
        return self.decode(ids)
    def decode_tgt(self, ids):
        return self.decode(ids)
    
    def validate_prediction(self, numerical_sequence):
        # there are no constraints
        return True
    
    def prediction_is_complete(self, numerical_sequence):
        return self.EOS in numerical_sequence
    
    def decode_tensor(self, output_tensor):
        """
        output_tensor: [[int]]: model output (Seq_len x batch)
        """
        batch_first_output_tensor = output_tensor.T
        return [self.decode(sequence.cpu().tolist()) for sequence in batch_first_output_tensor]
        
    def to_dataloader(self, batch_size, repeat=False, num_workers=4, shuffle=True):
        """
        This function returns an iterable object with all the data batched.
        
        >>> BPE_processor = CodeTrainedBPE_Translation_DataProcessor(validation_pairs, max_tgt_len=100)
        >>> dataloader = BPE_processor.to_dataloader(2)
        
        >>> for i_batch, sample_batched in enumerate(dataloader):
        >>>     print(sample_batched["tgt"])
        >>>     print(BPE_processor.decode_tensor(sample_batched["tgt"]))
        >>>     break
        """
        return DataLoader(self, batch_size=batch_size, num_workers=num_workers,\
                           drop_last=False, collate_fn = self.collate, shuffle=shuffle)
    
    def save(self, path):
        torch.save(self, path)
    
class Parse_Tree_Translation_DataProcessor(Dataset):
    def __init__(self, task_data, max_length=500, tokenizer_dir="/nfs/phd_by_carlos/notebooks/datasets/code_search_net/", 
                 grammar_path="src/tree-sitter/tree-sitter-python/src/grammar.json",
                 **kwargs):
        self.task_data = task_data
        self.max_length = max_length
        self.tokenizer = ByteLevelBPETokenizer(tokenizer_dir+"code_bpe_hugging_32k-vocab.json",
                                          tokenizer_dir+"code_bpe_hugging_32k-merges.txt")
        self.tokenizer.add_special_tokens(["[CLS]", "[SOS]", "[EOS]", "[PAD]"])
        self.SOS = self.tokenizer.encode("[SOS]").ids[0]
        self.EOS = self.tokenizer.encode("[EOS]").ids[0]
        self.PAD = self.tokenizer.encode("[PAD]").ids[0]
        self.CLS = self.tokenizer.encode("[CLS]").ids[0]
        
        with open(grammar_path, "r") as grammar_file:
            self.python_grammar = json.load(grammar_file)

        extra_externals = {"_string_start":{
                              "type": "PATTERN",
                              "value": '"'
                            },
                           "_string_content":{
                              "type": "PATTERN",
                              "value": "[A-Za-z0-9 _,.()\/{}!$@'*]*"
                            },
                           "_string_end":{
                              "type": "PATTERN",
                              "value": '"'
                            },
                           "_newline":{
                              "type": "BLANK"
                            }
                          }
        for node_type, member in extra_externals.items():
            self.python_grammar["rules"][node_type] = member

        self.python_parser = Code_Parser(self.python_grammar, "python", **kwargs)
        self.node_processor = Node_Processor()
        self.tree_vocab, grammar_patterns = get_grammar_vocab(self.python_grammar)
        
        self.tokenizer.add_tokens(["<REDUCE>"])
        for tree_token in sorted(self.tree_vocab):
            if len(self.tokenizer.encode(tree_token).tokens) != 1:
                self.tokenizer.add_tokens([tree_token])
                
        # filtering the data
        filtered_task_data = []
        for desc, code in self.task_data:
            numerical_code_sequence = self.encode_tgt(code)
            numerical_desc_sequence = self.encode_src(desc)
            token_sequence = self.numerical_to_token_sequence(numerical_code_sequence)
            if self.python_parser.is_valid_sequence(token_sequence) and len(token_sequence) <= max_length and len(numerical_desc_sequence) <= max_length:
                filtered_task_data.append((desc, code))
            elif len(token_sequence) > max_length or len(numerical_desc_sequence) > max_length:
                print(f"Sequence too long: src->{len(numerical_desc_sequence)}, tgt->{len(token_sequence)}")
            else:    
                print(f"Could not parse and reconstruct: {code}")
        self.task_data = filtered_task_data
    
    def __len__(self):
        return len(self.task_data)
    
    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError
            
        src, tgt = self.task_data[idx]
        sample = {'src': self.encode_src(src), 'tgt': self.encode_tgt(tgt)}
        return sample
    
    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

    
    def encode_src(self, desc_str):
        return [self.SOS] + self.tokenizer.encode(desc_str).ids + [self.EOS]
    
    def encode_tgt(self, code_str):
        code_sequence = self.python_parser.code_to_sequence(code_str)
        numerical_code = []
        for code_token in code_sequence:
            numerical_code += self.tokenizer.encode(code_token).ids
        return [self.SOS] + numerical_code + [self.EOS]
    
    def decode_src(self, numerical_desc):
        """
        ids: [int]: ids to decode
        """
        return self.tokenizer.decode(ids)
    
    def numerical_to_token_sequence(self, numerical_code):
        token_sequence = [self.tokenizer.decode([token_idx]) for token_idx in numerical_code if token_idx not in [self.SOS,self.EOS,self.PAD,self.CLS]]
        return token_sequence
    
    def decode_tgt(self, numerical_code):
        token_sequence = self.numerical_to_token_sequence(numerical_code)
        partial_tree = self.python_parser.sequence_to_partial_tree(token_sequence)
        return self.node_processor.pretty_print(partial_tree.root), partial_tree
    
    def validate_prediction(self, current_prediction):
#         print(f"validating: {current_prediction}")
        token_sequence = self.numerical_to_token_sequence(current_prediction)
        return self.python_parser.is_valid_sequence(token_sequence)
    
    def prediction_is_complete(self, current_prediction):
        token_sequence = self.numerical_to_token_sequence(current_prediction)
        return self.python_parser.sequence_to_partial_tree(token_sequence).is_complete
    
    def collate(self, input_samples):
        """
        input_samples: [dict]: these are samples obtained through the _get_item method
        """
        collated_samples = {}
        sample_keys = input_samples[0].keys()
        for key in sample_keys:
            collated_samples[key] = torch.nn.utils.rnn.pad_sequence([torch.Tensor(sample[key]).type(torch.LongTensor) for sample in input_samples], 
                                                 padding_value=self.PAD)
        return collated_samples
    
    def to_dataloader(self, batch_size, num_workers=4, shuffle=True):
        """
        This function returns an iterable object with all the data batched.
        
        >>> BPE_processor = CodeTrainedBPE_Translation_DataProcessor(validation_pairs, max_tgt_len=100)
        >>> dataloader = BPE_processor.to_dataloader(2)
        
        >>> for i_batch, sample_batched in enumerate(dataloader):
        >>>     print(sample_batched["tgt"])
        >>>     print(BPE_processor.decode_tensor(sample_batched["tgt"]))
        >>>     break
        """
        return DataLoader(self, batch_size=batch_size, num_workers=num_workers,\
                           drop_last=False, collate_fn = self.collate, shuffle=shuffle)
    
    def save(self, path):
        torch.save(self, path)
        
        