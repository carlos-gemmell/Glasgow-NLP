from .file_ops import corpus_to_array
from .useful_utils import filter_corpus, clean_samples, string_split_v3, jsonl_dir_to_data, download_from_url
from .models_and_transforms.text_transforms import Rename_Transform

from dotmap import DotMap
import numpy as np
import random
import torch
import json
import ujson
import sys
import pickle
from pyserini.search import SimpleSearcher
from .metrics import RecipRank, doc_search_subtask
import csv
import json_lines
import os
from tqdm import tqdm 

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
    def __init__(self, data_directory="./datasets/code_search_net", language="python", random_shuffle_seed=-1, max_chars=2000):
        """
        >>> from src.dataset_loaders import CodeSearchNet_RawDataLoader
        >>> codeSearchNet_data_loader = CodeSearchNet_RawDataLoader()
        >>> validation_pairs = codeSearchNet_data_loader.english_to_code_for_translation("valid")
        """
        self.data_directory = data_directory
        if not os.path.exists(data_directory):
            os.makedirs(data_directory)
        
        collection_dir = os.path.join(data_directory, f"{language}.zip")
        zip_file = os.path.join(data_directory, f"{language}.zip")
        if not os.path.isfile(zip_file):
            download_from_url(f"https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/{language}.zip", zip_file)
            os.system(f'unzip {zip_file} -d {data_directory}')
        
        lang_data_dir = os.path.join(data_directory, language, "final/jsonl")
        self.train_split = jsonl_dir_to_data(os.path.join(lang_data_dir, "train")) # these are all arrays of dicts
        self.valid_split = jsonl_dir_to_data(os.path.join(lang_data_dir, "valid"))
        self.test_split = jsonl_dir_to_data(os.path.join(lang_data_dir, "test"))
        self.max_chars = max_chars
        
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
            return self.train_split + self.valid_split
        else:
            raise Exception(f"'{split}' split not recognised.")
            
    def get_samples(self, split, fields=["original_string"]):
        '''
        split: str: "train", "test", "valild", "all"
        fields: [str]: ["original_string", 'code', 'code_tokens', 'docstring', 'docstring_tokens', 'repo', 'url', 'func_name', 'path']
                       specify the fields returned in each sample
        '''
        samples = self._get_split(split)
        samples = [sample_obj for sample_obj in tqdm(samples, desc=f'Filtering max chars: {self.max_chars}') if len(sample_obj['original_string'])<self.max_chars]
        new_samples = []
        for sample_obj in tqdm(samples, desc='deleting bload fields'):
            new_sample = {}
            for key in list(sample_obj.keys()):
                if key in fields and fields:
                    new_sample[key] = sample_obj[key]
            new_sample['only_code'] = sample_obj['code'].replace(sample_obj['docstring'], '')
            new_samples.append(new_sample)
        return new_samples       
        
class CoNaLa_RawDataLoader(RawDataLoader):
    def __init__(self, data_directory="/nfs/phd_by_carlos/notebooks/datasets/CoNaLa/"):
        self.data_directory = data_directory
        
        with open(os.path.join(data_directory, "conala-train.json")) as train_f:
            self.full_train_data = json.load(train_f)
            self.train_data = self.full_train_data[:int(len(self.full_train_data)*0.9)]
            self.valid_data = self.full_train_data[int(len(self.full_train_data)*0.9):]
        with open(os.path.join(data_directory, "conala-test.json")) as train_f:
            self.test_data = json.load(train_f)
        if not os.path.isfile(os.path.join(data_directory, 'GPT3/GPT3_CoNaLa_mined_solid_state_501784_no_probs.json')):
            download_from_url(f"https://storage.googleapis.com/carlos-phd-data/GPT3_CoNaLa_mined_solid_state_501784_no_probs.json",
                              os.path.join(data_directory, 'GPT3/GPT3_CoNaLa_mined_solid_state_501784_no_probs.json'))
        with open(os.path.join(data_directory, 'GPT3/GPT3_CoNaLa_mined_solid_state_501784_no_probs.json')) as mined_f:
            self.mined_samples = json.load(mined_f)
            
    def _get_split(self, split):
        if split == "train":
            return self.train_data
        elif split == "valid":
            return self.valid_data
        elif split == "test":
            return  self.test_data
        elif split == "all":
            return self.full_train_data
        elif split == "mined_GPT3":
            return Rename_Transform(fields=[('code', 'snippet'), ('GPT3_pred_desc','rewritten_intent')])(self.mined_samples[:])
        else:
            raise Exception(f"'{split}' split not recognised.")
            
    def get_samples(self, split, max_char_len=600, min_char_len=3):
        """
        This method returns parallel samples of English and Code for the CoNaLa dataset.
        split: str: "train", "valid", "test", "all"
        
        returns: [dict]: [{'description':"desc text", 'code': "code text"}]
        """
        data_split = self._get_split(split)
        samples = []
        for sample in data_split:
            if "rewritten_intent" in sample and "snippet" in sample:
                desc = sample["rewritten_intent"] if sample["rewritten_intent"] != None else sample["intent"]
                code = sample["snippet"]
                if min_char_len<len(desc)<max_char_len and min_char_len<len(code)<max_char_len:
                    samples.append({'description':desc, 'code':code})
        return samples
    
class Django_RawDataLoader(RawDataLoader):
    def __init__(self, data_directory="/nfs/phd_by_carlos/notebooks/datasets/django_folds/", **kwargs):
        self.data_directory = data_directory
        
        def get_lines(path):
            lines = []
            with open(path) as f:
                for line in f.readlines():
                    lines.append(line.strip())
            return lines
        train_src = get_lines(os.path.join(data_directory, f"django.fold1-10.train.src"))
        train_tgt = get_lines(os.path.join(data_directory, f"django.fold1-10.train.tgt"))
        valid_src = get_lines(os.path.join(data_directory, f"django.fold1-10.valid.src"))
        valid_tgt = get_lines(os.path.join(data_directory, f"django.fold1-10.valid.tgt"))
        test_src = get_lines(os.path.join(data_directory, f"django.fold1-10.test.src"))
        test_tgt = get_lines(os.path.join(data_directory, f"django.fold1-10.test.tgt"))
        
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
    
    def get_samples(self, split):
        """
        This method returns parallel samples of English and Code for the Django dataset.
        split: str: "train", "valid", "test", "all"
        
        returns: [dict]: [{'description':"desc text", 'code': "code text"}]
        """
        data_split = self._get_split(split)
        samples = []
        for desc, code in data_split:
            samples.append({'description':desc, 'code':code})
        return samples
    
    
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
#                     if q_id not in self.q_rels:
#                         continue
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
    
    def get_topics(self, split, ignore_missing_q_rels=False):
        '''
        split: str: "train", "dev", "all", "eval"
        returns: [dict]: [{'q_id':"32_4", 'q_rel':["CAR_xxx",..]}, 'prev_turns':["32_3",..],...]
        '''
        topic_split = self.get_split(split)
        samples = []
        samples = [{'prev_turns':prev_turns, 'q_id':q_id, 'q_rel':self.q_rels.get(q_id)} 
                   for q_id, prev_turns in topic_split.items() if q_id in self.q_rels or ignore_missing_q_rels]
        return samples
    
    def get_query(self, q_id, utterance_type="raw_utterance"):
        '''
        >>> raw_data_loader.get_query("31_4", utterance_type="manual_rewritten_utterance")
        '''
        return self.query_collection[q_id][utterance_type]
    
    
class Java_Small_RawDataLoader():
    def __init__(self, data_directory="/nfs/phd_by_carlos/notebooks/datasets/java/", max_train_samples=10000):
        
        self.data_directory = data_directory
        if not os.path.exists(data_directory):
            os.makedirs(data_directory)
            
        train_file = os.path.join(data_directory, f"java-small-json/java-small.train.json")
        valid_file = os.path.join(data_directory, f"java-small-json/java-small.val.json")
        test_file = os.path.join(data_directory, f"java-small-json/java-small.test.json")
        
        train_samples_file = os.path.join(data_directory, f"java-small-json/train_samples.json")
        valid_samples_file = os.path.join(data_directory, f"java-small-json/val_samples.json")
        test_samples_file = os.path.join(data_directory, f"java-small-json/test_samples.json")
        
        zip_file = os.path.join(data_directory, f"java-small-json.tar.gz")
        if not os.path.isfile(zip_file):
            download_from_url(f"https://codegen-slm.s3.us-east-2.amazonaws.com/data/java-small-json.tar.gz", zip_file)
            os.system(f'tar -xvzf {zip_file} -C {data_directory} --no-same-owner')
            
            print("One time load operation, cleaning up tree from data")
            with open(train_file, 'r') as train_f, open(valid_file, 'r') as valid_f, open(test_file, 'r') as test_f:
                train_samples = []
                train_reader = json_lines.reader(train_f)
                num_lines = sum(1 for line in open(train_file))
                for i, sample in tqdm(enumerate(train_reader), desc="loading train", total=num_lines):
                    code = sample['left_context']+sample['target_seq']+sample['right_context']
                    train_samples.append({'code':code})
                    if i > max_train_samples:
                        break
                ujson.dump(train_samples, open(train_samples_file, 'w'))

                valid_samples = []
                valid_reader = json_lines.reader(valid_f)
                num_lines = sum(1 for line in open(valid_file))
                for sample in tqdm(valid_reader, desc="loading valid", total=num_lines):
                    code = sample['left_context']+sample['target_seq']+sample['right_context']
                    valid_samples.append({'code':code})
                ujson.dump(valid_samples, open(valid_samples_file, 'w'))

                test_samples = []
                test_reader = json_lines.reader(test_f)
                num_lines = sum(1 for line in open(test_file))
                for sample in tqdm(test_reader, desc="loading test", total=num_lines):
                    code = sample['left_context']+sample['target_seq']+sample['right_context']
                    test_samples.append({'code':code})
                ujson.dump(test_samples, open(test_samples_file, 'w'))
        
        print("Loading simplified train/val/test files")
        self.train_samples = ujson.load(open(train_samples_file, 'r'))[:max_train_samples]
        self.valid_samples = ujson.load(open(valid_samples_file, 'r'))
        self.test_samples = ujson.load(open(test_samples_file, 'r'))
        
    def _get_split(self, split):
        if split=="train":
            return self.train_samples
        elif split=="valid":
            return self.valid_samples
        elif split=="test":
            return self.test_samples
        elif split=="all":
            return self.train_samples + self.valid_samples
        else:
            raise Exception(f"'{split}' split not recognised.")
            
    def get_samples(self, split):
        '''
        split: str: "train", "test", "valild", "all"
        returns: [dict]: [{'code':"java code"},...]
        '''
        samples = self._get_split(split)
        return samples