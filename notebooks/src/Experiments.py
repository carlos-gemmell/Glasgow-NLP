from abc import ABC, abstractmethod
import torch
from .metrics import nltk_bleu
import numpy as np
import os
import sys
from .useful_utils import string_split_v3, string_split_v1, chunks
import pytrec_eval
import json
import re
from tqdm.auto import tqdm 
from .bleu_score import compute_bleu

class Experiment(ABC):
    def __init__(self, task_data):
        """
        task_data: [(str, str)]: input/target pairs for translation evaluation.
        """
        self.task_data = task_data
    
    @abstractmethod
    def evaluate(self, prediction_fn):
        """
        This function should compute all relevant metrics to the task,
        prediction_fn: (inp) -> (pred): it's an end-to-end prediction function from any model.
        
        returns: dict: metrics
        """
        pass
        
    def save(self, path):
        """
        Saves the entire object ready to be loaded.
        """
        torch.save(self, path)
        
    def load(path):
        """
        STATIC METHOD
        accessed through class, loads a pre-existing experiment.
        """
        return torch.load(path)
    
    
class TranslationExperiment(Experiment):
    def __init__(self, task_data, src_splitter=string_split_v1, tgt_splitter=string_split_v1):
        """
        task_data: [(str, str)]: this is the expected data format.
        
        >>> from src.Experiments import TranslationExperiment
        >>> translation_experiment = TranslationExperiment(validation_pairs)
        >>> def simple_translate(src):
        >>>     return "return output"
        >>> translation_experiment.evaluate(simple_translate)
            {'BLEU': 1.4384882092392364e-09}
        """
        super().__init__(task_data)
        self.src_splitter = src_splitter
        self.tgt_splitter = tgt_splitter
        
        
    def evaluate(self, prediction_fn, save_dir=None, save_name="translation_eval.txt", batched=None):
        """
        Produces evaluation scores and saves the results to a file. The tokenisation is done through string_split_v1.
        So any non spaced text will be considered as one token.
        
        prediction_fn: (str)->(str) or [str]->[str]
        save_dir: str: folder to save the file
        save_name: str: name of file
        batched: int or None: size to use for the prediction function
        """
        if batched:
            src_sents = [src for (src, tgt) in self.task_data]
            chunked_sents = list(chunks(src_sents, batched))
            predictions = [prediction_fn(sents) for sents in tqdm.tqdm(chunked_sents, desc="predicting", total=len(chunked_sents))]
            predictions = [val for sublist in predictions for val in sublist] # flattening
        else:
            predictions = [prediction_fn(src) for (src, tgt) in tqdm.tqdm(self.task_data, desc="predicting")]
        
        # BLEU calculation
        BLEU_scores = []
        for (src, tgt), pred in tqdm.tqdm(list(zip(self.task_data, predictions)), desc="calculating bleu"):
            BLEU_score = nltk_bleu(self.tgt_splitter(tgt), self.tgt_splitter(pred))
            BLEU_scores.append(BLEU_score)
        total_BLEU = np.average(BLEU_scores)
            
        # Write to file
        if save_dir != None:
            save_path = os.path.join(save_dir, save_name)
            print(f"saving translation eval to file: {save_path}")
            with open(save_path, "w", encoding="utf-8") as out_fp:
                for (src, tgt), pred, BLEU in zip(self.task_data, predictions, BLEU_scores):

                    out_fp.write("SRC  :" + src + "\n")
                    out_fp.write("TGT  :" + tgt + "\n")
                    out_fp.write("PRED :" + pred + "\n")
                    out_fp.write("BLEU :" + str(BLEU) + "\n")
                    out_fp.write("\n")

                out_fp.write("\n\n| EVALUATION | BLEU: {:5.2f} |\n".format(total_BLEU))
                print("| EVALUATION | BLEU: {:5.3f} |".format(total_BLEU))
            
        return {"BLEU":total_BLEU}
            
        
        
class CAsT_experiment(Experiment):
    def __init__(self, topics):
        '''
        topics: (context:[q_ids], q_id, q_rel:[d_ids])
        '''
        self.topics = topics
        
    def evaluate(self, prediction_fn, save_dir=None, save_name="translation_eval.txt", hits=100):
        full_q_rels = {}
        run = {}
        for topic in self.topics:
            pred_d_ids = prediction_fn(topic, hits=100)
            context, q_id, q_rels = topic
            full_q_rels[q_id] = {d_id:1 for d_id in q_rels}
            run[q_id] = {d_id:score for (d_id, score) in pred_d_ids}
        evaluator = pytrec_eval.RelevanceEvaluator(full_q_rels, {'map', 'ndcg'})
        results = evaluator.evaluate(run)
        aggregate = self.dict_mean(list(results.values()))
        return aggregate, results
    
    def dict_mean(self, dict_list):
        mean_dict = {}
        for key in dict_list[0].keys():
            mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
        return mean_dict
    
    

class Ranking_Experiment():
    def __init__(self, q_rels, save_dir=None, save_name="rerank_eval.run"):
        '''
        q_rels: dict: {'q_id':[d_id, d_id,...],...}
        '''
        pytrec_q_rels = {}
        for q_id, d_ids in q_rels.items():
            pytrec_q_rels[q_id] = {d_id:1 for d_id in d_ids}
        self.evaluator = pytrec_eval.RelevanceEvaluator(pytrec_q_rels, {'map', 'ndcg_cut_3', 'set_recall', 'recip_rank'})
        
    def dict_mean(self, dict_list):
        mean_dict = {}
        for key in dict_list[0].keys():
            mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
        return mean_dict
    
    def __call__(self, samples):
        '''
        samples: [dict]: [{'q_id':"xxx", 'search_results':[("MARCO_xxx", 0.63)...]},...]
        '''
        pytrec_run = {}
        for sample_obj in samples:
            q_id = sample_obj['q_id']
            pytrec_run[q_id] = {}
            for d_id, score in sample_obj['search_results']:
                pytrec_run[q_id][d_id] = score
        
        results = self.evaluator.evaluate(pytrec_run)
        for sample_obj, result in zip(samples, results.values()):
            sample_obj.update(result)
        aggregate = self.dict_mean(list(results.values()))
        return aggregate
    
class Sequence_BLEU_Experiment():
    def __init__(self, fields={}, debug=True):
        '''
        An Experiment to evaluate sequence similarity through metrics like: BLEU or token accuracy.
        '''
        self.fields = {'predicted_seq':'predicted_seq', 'target_seq':'target_seq'}
        self.debug = debug
        self.fields.update(fields)
    
    def __call__(self, samples):
        '''
        samples: [dict]: [{'target_seq':"taget text", 'predicted_seq':"pred text"},...]
        returns: [dict]: [{'target_seq':"taget text", 'predicted_seq':"pred text", "BELU":0.6},...]
        '''
        for sample_obj in samples:
            pred_tokens = self.tokenize_for_bleu_eval(sample_obj[self.fields['predicted_seq']])
            refrence_tokens = self.tokenize_for_bleu_eval(sample_obj[self.fields['target_seq']])
            if pred_tokens==[]:
                pred_tokens = ['']
            sample_obj["nltk_BLEU"] = nltk_bleu(refrence_tokens, pred_tokens)
            
        if self.debug:
            corpus_bleu = compute_bleu([[self.tokenize_for_bleu_eval(s[self.fields['target_seq']])] for s in samples], [self.tokenize_for_bleu_eval(s[self.fields['predicted_seq']]) for s in samples], smooth=False)[0]
            nltk_BLEU = np.average([s["nltk_BLEU"] for s in samples])
            print(f'corpus_official_BLEU: {corpus_bleu}')
            print(f'nltk_BLEU: {nltk_BLEU}')
        
        return samples
    
    def tokenize_for_bleu_eval(self, code):
        """ 
        The tokenizer that we use for code submissions, from Wang Ling et al., Latent Predictor Networks for Code Generation (2016)
        @param code: string containing a code snippet
        @return: list of code tokens
        """
        code = re.sub(r'([^A-Za-z0-9_])', r' \1 ', code)
        code = re.sub(r'([a-z])([A-Z])', r'\1 \2', code)
        code = re.sub(r'\s+', ' ', code)
        code = code.replace('"', '`')
        code = code.replace('\'', '`')
        tokens = [t for t in code.split(' ') if t]

        return tokens
    
class RUN_File_Transform_Exporter():
    def __init__(self, run_file_path, model_name='model_by_carlos'):
        '''
        A Transform Exporter that creates a RUN file from samples returnedd by a search engine.
        '''
        self.run_file_path = run_file_path
        self.model_name = model_name
        
    def __call__(self, samples):
        '''
        samples: [dict]: [{'q_id':"xxx", 'search_results':[("MARCO_xxx", 0.63)...]},...]
        '''
        total_samples = 0
        with open(self.run_file_path, 'w') as run_file:
            for sample_obj in tqdm(samples, desc='Writing to RUN file', leave=False):
                q_id = sample_obj['q_id']
                search_results = sample_obj['search_results']
                ordered_results = sorted(search_results, key=lambda res: res[1], reverse=True)
                for idx, result in enumerate(ordered_results):
                    d_id, score = result
                    total_samples+=1
                    run_file.write(f"{q_id} Q0 {d_id} {idx+1} {score} {self.model_name}\n")
        print(f"Successfully written {total_samples} samples from {len(samples)} queries run to: {self.run_file_path}")