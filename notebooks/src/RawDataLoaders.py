from .file_ops import corpus_to_array
from .useful_utils import filter_corpus, clean_samples, string_split_v3, jsonl_dir_to_data, download_from_url

from torchtext.data import Field, BucketIterator
import torchtext
from dotmap import DotMap
import numpy as np
import random
import torch
import json
import sys
import pickle
from pyserini.search import SimpleSearcher
from .metrics import RecipRank, doc_search_subtask
import csv
import os
from tqdm.auto import tqdm 

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
    def __init__(self, data_directory="/nfs/phd_by_carlos/notebooks/datasets/django/", **kwargs):
        self.data_directory = data_directory
        
        def get_lines(path):
            lines = []
            with open(path) as f:
                for line in f.readlines():
                    lines.append(line.strip())
            return lines
        train_src = get_lines(os.path.join(data_directory, f"train.anno"))
        train_tgt = get_lines(os.path.join(data_directory, f"train.code"))
        valid_src = get_lines(os.path.join(data_directory, f"dev.anno"))
        valid_tgt = get_lines(os.path.join(data_directory, f"dev.code"))
        test_src = get_lines(os.path.join(data_directory, f"test.anno"))
        test_tgt = get_lines(os.path.join(data_directory, f"test.code"))
        
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
    def __init__(self, **kwargs):
        from src.DataProcessors import Parse_Tree_Translation_DataProcessor
        super().__init__(**kwargs)
        
        parse_tree_processor = Parse_Tree_Translation_DataProcessor(self.train_pairs, **kwargs)
        print("Flitered train ratio:",len(self.train_pairs), len(parse_tree_processor.task_data))
        self.train_pairs = parse_tree_processor.task_data
        
        parse_tree_processor = Parse_Tree_Translation_DataProcessor(self.valid_pairs, **kwargs)
        print("Flitered valid ratio:",len(self.valid_pairs), len(parse_tree_processor.task_data))
        self.valid_pairs = parse_tree_processor.task_data
        
        parse_tree_processor = Parse_Tree_Translation_DataProcessor(self.test_pairs, **kwargs)
        print("Flitered test ratio:",len(self.test_pairs), len(parse_tree_processor.task_data))
        self.test_pairs = parse_tree_processor.task_data
        
class CAR_RawDataLoader(RawDataLoader):
    def __init__(self, data_directory="/nfs/phd_by_carlos/notebooks/datasets/TREC_CAR/", **kwargs):
        self.data_directory = data_directory
        if not os.path.exists(data_directory):
            os.makedirs(data_directory)
            
        if not os.path.isfile(collection_dir):    
            pass

class MS_Marco_RawDataLoader(RawDataLoader):
    def __init__(self, data_directory="/nfs/phd_by_carlos/notebooks/datasets/MS_MARCO/", **kwargs):
        self.data_directory = data_directory
        if not os.path.exists(data_directory):
            os.makedirs(data_directory)
        
        collection_dir = os.path.join(data_directory, "collection.tsv")
        if not os.path.isfile(collection_dir):
            gzip_file = os.path.join(data_directory, "collection.tsv.gz")
            download_from_url("https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz", gzip_file)
            os.system(f'gzip -dk {gzip_file}')
        
        if os.path.isfile(os.path.join(data_directory, "collection.pickle")):
            with open(os.path.join(data_directory, "collection.pickle"), "rb") as f:
                # pickle.dump(raw_data_loader.collection, open("datasets/MS_MARCO/collection.pickle", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
                self.collection = pickle.load(f)
        else:
            self.collection = {}
            with open(os.path.join(data_directory, "collection.tsv")) as tsvfile:
                reader = csv.reader(tsvfile, delimiter='\t')
                for i, row in tqdm(enumerate(reader), total=8841823, desc="loading docs", leave=False):
                    passage_id, passage_text = row
                    self.collection[f"MARCO_{int(passage_id)}"] = passage_text
                pickle.dump(self.collection, open(os.path.join(data_directory, "collection.pickle"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        
        queries_zip_file = os.path.join(data_directory, "queries.tar.gz")
        if not os.path.isfile(queries_zip_file):
            download_from_url("https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz", queries_zip_file)
            os.system(f'tar -xvzf {queries_zip_file} -C {data_directory}')
        
        splits = ["train","dev","eval"]
        self.queries = {}
        for split in splits:
            self.queries.update(self.load_queries(split))
                
    def load_queries(self, split):
        query_dict = {}                     
        with open(os.path.join(self.data_directory, f"queries.{split}.tsv")) as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            for i, row in tqdm(enumerate(reader), desc=f"loading {split} queries", leave=False):
                q_id, q_text = row
                query_dict[q_id] = q_text
        return query_dict
    
    def get_doc(self, d_id, **kwargs):
        return self.collection[d_id]
    def get_query(self, q_id, **kwargs):
        return self.queries[q_id]
    
    def get_topics(self, split):
        '''
        split: str: "train", "dev", "eval"
        returns: [dict]: [{'q_id':"32_4", 'q_rel':["MARCO_xxx",..]},...]
        '''
        q_ids = list(self.load_queries(split).keys())
        q_rels = self.q_rels(split)
        return [{'q_id':q_id, 'q_rel':q_rels[q_id]} for q_id in q_ids if q_id in q_rels]
    
    def q_rels(self, split):
        '''
        returns: dict: {'q_id':[d_id, d_id,...],...}
        '''
        q_rels_dict = {}
        q_rels_file = os.path.join(self.data_directory, f"q_rels.{split}.tsv")
        if not os.path.isfile(q_rels_file):
            download_from_url(f"https://msmarco.blob.core.windows.net/msmarcoranking/qrels.{split}.tsv", q_rels_file)
                                     
        with open(q_rels_file) as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            for i, row in tqdm(enumerate(reader), desc="loading q_rels", leave=False):
                q_id, _, passage_id, _ = row
                if int(q_id) in q_rels_dict:
                    q_rels_dict[q_id].append(int(passage_id))
                else:
                    q_rels_dict[q_id] = [f"MARCO_{int(passage_id)}"]
        return q_rels_dict
    

class CAsT_RawDataLoader():
    def __init__(self, CAsT_index="/nfs/phd_by_carlos/notebooks/datasets/TREC_CAsT/CAsT_collection_with_meta.index",
                 treccastweb_dir="/nfs/phd_by_carlos/notebooks/datasets/TREC_CAsT/treccastweb",
                 NIST_qrels="/nfs/phd_by_carlos/notebooks/datasets/TREC_CAsT/2019qrels.txt", **kwargs):
        
        self.searcher = SimpleSearcher(CAsT_index)
        self.q_rels = {}
        with open(NIST_qrels) as NIST_fp:
            for line in NIST_fp.readlines():
                q_id, _, d_id, score = line.split(" ")
                if int(score) < 3:
                    # ignore some of the worst ranked
                    continue
                if q_id not in self.q_rels:
                    self.q_rels[q_id] = []
                self.q_rels[q_id].append(d_id)
                
        with open(os.path.join(treccastweb_dir,"2020/2020_manual_evaluation_topics_v1.0.json")) as y2_fp:
            y2_data = json.load(y2_fp)
            for topic in y2_data:
                topic_id = topic["number"]
                for turn in topic["turn"]:
                    turn_id = turn["number"]
                    q_id = f"{topic_id}_{turn_id}"
                    if q_id not in self.q_rels:
                        self.q_rels[q_id] = []
                    self.q_rels[q_id].append(turn["manual_canonical_result_id"])
        
        year1_query_collection, self.year1_topics = self.load_CAsT_topics_file(os.path.join(treccastweb_dir,"2019/data/evaluation/evaluation_topics_v1.0.json"))
        year2_query_collection, self.year2_topics = self.load_CAsT_topics_file(os.path.join(treccastweb_dir,"2020/2020_manual_evaluation_topics_v1.0.json"))
        
        self.query_collection = {**year1_query_collection, **year2_query_collection}
        
        with open(os.path.join(treccastweb_dir, "2019/data/evaluation/evaluation_topics_annotated_resolved_v1.0.tsv")) as resolved_f:
            reader = csv.reader(resolved_f, delimiter="\t")
            for row in reader:
                q_id, resolved_query = row
                if q_id in self.query_collection:
                    self.query_collection[q_id]["manual_rewritten_utterance"] = resolved_query
    
    def NIST_result_curve(self, score):
        "0->0, 1~>0.1, 3~>0.6, 4->1"
        return (1/16)*(score**2)
        
    def load_CAsT_topics_file(self, file):
        query_collection = {}
        topics = {}
        with open(file) as topics_fp:
            topics_data = json.load(topics_fp)
            for topic in topics_data:
                previous_turns = []
                topic_id = topic["number"]
                for turn in topic["turn"]:
                    turn_id = turn["number"]
                    q_id = f"{topic_id}_{turn_id}"
                    if q_id not in self.q_rels:
                        continue
                    query_collection[q_id] = turn
                    topics[q_id] = previous_turns[:]
                    previous_turns.append(q_id)
            return query_collection, topics
        
        
       
    def get_doc(self, doc_id):
        raw_text = self.searcher.doc(doc_id).raw()
        paragraph = raw_text[raw_text.find('<BODY>\n')+7:raw_text.find('\n</BODY>')]
        return paragraph
    
    def get_split(self, split):
        '''
        return: dict: {q_id: [q_id,...]}: contains the query turns and the previous query turns for the topic
        '''
        dev_topic_cutoff = 71
        if split == "train":
            return {k:v for k,v in self.year1_topics.items() if int(k.split("_")[0]) < dev_topic_cutoff}
        elif split == "dev":
            return {k:v for k,v in self.year1_topics.items() if int(k.split("_")[0]) >= dev_topic_cutoff}
        elif split == "all":
            return self.year1_topics
        elif split == "eval":
            return self.year2_topics
        else:
            raise Exception(f"Split '{split}' not recognised")
    
    def get_topics(self, split):
        '''
        split: str: "train", "dev", "all", "eval"
        returns: [dict]: [{'q_id':"32_4", 'q_rel':["CAR_xxx",..]}, 'prev_turns':["32_3",..],...]
        '''
        topic_split = self.get_split(split)
        samples = [{'prev_turns':prev_turns, 'q_id':q_id, 'q_rel':self.q_rels[q_id]} for q_id, prev_turns in topic_split.items()]
        return samples
    
    def get_query(self, q_id, utterance_type="raw_utterance"):
        '''
        >>> raw_data_loader.get_query("31_4", utterance_type="manual_rewritten_utterance")
        '''
        return self.query_collection[q_id][utterance_type]