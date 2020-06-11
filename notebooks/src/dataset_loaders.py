from .file_ops import corpus_to_array
from .useful_utils import filter_corpus, clean_samples, string_split_v3, jsonl_dir_to_data

from torchtext.data import Field, BucketIterator
import torchtext
from dotmap import DotMap
import numpy as np
import random
import json
import sys
from .metrics import RecipRank, doc_search_subtask
import csv
import tqdm
def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')
if is_interactive():
    import tqdm.notebook as tqdm 

import os
from abc import ABC, abstractmethod

class SRC_TGT_pairs():
    def __init__(self, src_fp, tgt_fp, max_seq_len=50, split_fn=string_split_v3):
        self.samples = corpus_to_array(src_fp, tgt_fp)
        self.samples = filter_corpus(self.samples, max_seq_len, tokenizer=split_fn)
        self.samples = clean_samples(self.samples)
        
    def get_samples(self):
        return self.samples
    

class DummyDataset():
    def __init__(self):
        pass
    
    def document_pairs(self, n):
        indepentent_samples = [
            "ham and cheese",
            "banana and toast",
            "olives and bread",
            "carrots and onions",
            "butter and jam",
            "beans and toast",
            "peppers and carrots"
        ]
        n_set = np.random.choice(indepentent_samples, n)
        return [("I like "+sample, "he likes "+sample) for sample in n_set]
    

class RawDataLoader(ABC):
    def __init__(self, data_directory, random_shuffle_seed=-1):
        """
        data_directory: str: this should point to the directory containing the file or files and could be used to store extra files
        random_shuffle_seed: int: a value to randomize a complete shuffle of the train, valid and test sets
        """
        super().__init__()
        
class CodeSearchNet_RawDataLoader(RawDataLoader):
    def __init__(self, data_directory="/nfs/code_search_net_archive", language="python", random_shuffle_seed=-1):
        """
        >>> from src.dataset_loaders import CodeSearchNet_RawDataLoader
        >>> codeSearchNet_data_loader = CodeSearchNet_RawDataLoader()
        >>> validation_pairs = codeSearchNet_data_loader.english_to_code_for_translation("valid")
        """
        lang_data_dir = os.path.join(data_directory, language, "final/jsonl")
        self.train_split = jsonl_dir_to_data(os.path.join(lang_data_dir, "train")) # these are all arrays of dicts
        self.valid_split = jsonl_dir_to_data(os.path.join(lang_data_dir, "valid"))
        self.test_split = jsonl_dir_to_data(os.path.join(lang_data_dir, "test"))
        
        if random_shuffle_seed != -1:
            raise Exception(f'random_shuffle_seed not implemented yet')
    
    def _get_split(self, split):
        if split=="train":
            return self.train_split
        elif split=="valid":
            return self.valid_split
        elif split=="test":
            return self.test_split
        elif split=="all":
            return self.train_split + self.valid_split + self.test_split
        else:
            raise Exception(f"'{split}' split not recognised.")
    
    def english_to_code_for_translation(self, split, size=sys.maxsize, full_docstring=False):
        """
        This method returns a standard format of [(source_str, target_str)] used for translation.
        
        split: str: "train", "valid", "test", "all"
        """
        data_split = self._get_split(split)
        if full_docstring:
            source_sents = [sample["docstring"] for sample in data_split]
        else:
            source_sents = [" ".join(sample["docstring_tokens"]) for sample in data_split]
        
        target_sents = [sample["code"].replace(sample["docstring"],"") for sample in data_split]
        
        translation_pairs = list(zip(source_sents,target_sents))
        return translation_pairs[:size]
    
    def code_to_english_for_translation(self, split, size=sys.maxsize, full_docstring=False):
        flipped_translation_pairs = self.english_to_code_for_translation(split, size=size, full_docstring=full_docstring)
        translation_pairs = [(code, eng) for (eng, code) in flipped_translation_pairs]
        return translation_pairs
    
    def english_to_code_for_search(self, split, size=sys.maxsize, full_docstring=False):
        """
        returns: ([str]*n, [str], [[int]]*n): queries, docs, mapping (q_rels) for each query (*n) 
                                              indicating which document idx is relevant.
        """
        translation_pairs = self.english_to_code_for_translation(split, size=size, full_docstring=full_docstring)
        queries = [eng for (eng, code) in translation_pairs]
        docs = [code for (eng, code) in translation_pairs]
        q_rels = [[i] for i in range(len(queries))]
        
        return queries, docs, q_rels
    
    def code_for_language_modeling(self, split, size=sys.maxsize, keep_docstring=False):
        """
        returns: [str]: an array of code samples as strings.
        """
        data_split = self._get_split(split)
        if keep_docstring:
            target_sents = [sample["code"] for sample in data_split]
        else:
            target_sents = [sample["code"].replace(sample["docstring"],"") for sample in data_split]
        return target_sents[:size]
        
        
class CoNaLa_RawDataLoader(RawDataLoader):
    def __init__(self, data_directory="/nfs/phd_by_carlos/notebooks/datasets/CoNaLa/"):
        self.data_directory = data_directory
        
        with open(os.path.join(data_directory, "conala-train.json")) as train_f:
            self.full_train_data = json.load(train_f)
            self.train_data = self.full_train_data[:int(len(self.full_train_data)*0.9)]
            self.valid_data = self.full_train_data[int(len(self.full_train_data)*0.9):]
        with open(os.path.join(data_directory, "conala-test.json")) as train_f:
            self.test_data = json.load(train_f)
            
    def _get_split(self, split):
        if split == "train":
            return self.train_data
        elif split == "valid":
            return self.valid_data
        elif split == "test":
            return  self.test_data
        elif split == "all":
            return self.full_train_data
        else:
            raise Exception(f"'{split}' split not recognised.")
            
    def english_to_code_for_translation(self, split):
        """
        This method returns a standard format of [(source_str, target_str)] used for translation.
        split: str: "train", "valid", "test", "all"
        """
        data_split = self._get_split(split)
        pairs = []
        for sample in data_split:
            if "rewritten_intent" in sample and "snippet" in sample:
                desc = sample["rewritten_intent"] if sample["rewritten_intent"] != None else sample["intent"]
                code = sample["snippet"]
                pairs.append((desc, code))
        return pairs
    
class Django_RawDataLoader(RawDataLoader):
    def __init__(self, data_directory="/nfs/phd_by_carlos/notebooks/datasets/django_folds/", fold=1):
        self.data_directory = data_directory
        
        def get_lines(path):
            lines = []
            with open(path) as f:
                for line in f.readlines():
                    lines.append(line.strip())
            return lines
        train_src = get_lines(os.path.join(data_directory, f"django.fold{fold}-10.train.src"))
        train_tgt = get_lines(os.path.join(data_directory, f"django.fold{fold}-10.train.tgt"))
        valid_src = get_lines(os.path.join(data_directory, f"django.fold{fold}-10.valid.src"))
        valid_tgt = get_lines(os.path.join(data_directory, f"django.fold{fold}-10.valid.tgt"))
        test_src = get_lines(os.path.join(data_directory, f"django.fold{fold}-10.test.src"))
        test_tgt = get_lines(os.path.join(data_directory, f"django.fold{fold}-10.test.tgt"))
        
        self.train_pairs = list(zip(train_src, train_tgt))
        self.valid_pairs = list(zip(valid_src, valid_tgt))
        self.test_pairs = list(zip(test_src, test_tgt))
        
    def _get_split(self, split):
        if split == "train":
            return self.train_pairs
        elif split == "valid":
            return self.valid_pairs
        elif split == "test":
            return  self.test_pairs
        elif split == "all":
            return self.train_pairs + self.valid_pairs
        else:
            raise Exception(f"'{split}' split not recognised.")
            
    def english_to_code_for_translation(self, split):
        """
        This method returns a standard format of [(source_str, target_str)] used for translation.
        split: str: "train", "valid", "test", "all"
        """
        data_split = self._get_split(split)
        return data_split
    
    
class Parseable_Django_RawDataLoader(Django_RawDataLoader):
    def __init__(self):
        from src.DataProcessors import Parse_Tree_Translation_DataProcessor
        super().__init__()
        
        parse_tree_processor = Parse_Tree_Translation_DataProcessor(self.train_pairs)
        print("Flitered train ratio:",len(self.train_pairs), len(parse_tree_processor.task_data))
        self.train_pairs = parse_tree_processor.task_data
        
        parse_tree_processor = Parse_Tree_Translation_DataProcessor(self.valid_pairs)
        print("Flitered valid ratio:",len(self.valid_pairs), len(parse_tree_processor.task_data))
        self.valid_pairs = parse_tree_processor.task_data
        
        parse_tree_processor = Parse_Tree_Translation_DataProcessor(self.test_pairs)
        print("Flitered test ratio:",len(self.test_pairs), len(parse_tree_processor.task_data))
        self.test_pairs = parse_tree_processor.task_data
        

class MS_Marco_RawDataLoader(RawDataLoader):
    def __init__(self, data_directory="/nfs/phd_by_carlos/notebooks/datasets/MS_Marco/"):
        self.data_directory = data_directory
        self.collection = {}
        with open(os.path.join(data_directory, "collection.tsv")) as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            for i, row in tqdm.tqdm(enumerate(reader)):
                passage_id, passage_text = row
                self.collection[int(passage_id)] = passage_text
        
        splits = ["train","dev","eval"]
        self.queries = {}
        for split in splits:
            self.queries.update(self.load_queries(split))
                
    def load_queries(self, split):
        query_dict = {}
        with open(os.path.join(self.data_directory, f"queries.{split}.tsv")) as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            for i, row in tqdm.tqdm(enumerate(reader)):
                q_id, q_text = row
                query_dict[int(q_id)] = q_text
        return query_dict
    
    def q_rels(self, split):
        q_rels_dict = {}
        with open(os.path.join(self.data_directory, f"qrels.{split}.tsv")) as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            for i, row in tqdm.tqdm(enumerate(reader)):
                q_id, _, passage_id, _ = row
                if int(q_id) in q_rels_dict:
                    q_rels_dict[int(q_id)].append(int(passage_id))
                else:
                    q_rels_dict[int(q_id)] = [int(passage_id)]
        return q_rels_dict
    