from src.models_and_transforms.BM25_models import BM25_Ranker
from src.models_and_transforms.BERT_models import BERT_Reranker
from src.models_and_transforms.run_file_models import Run_File_Searcher
from src.models_and_transforms.text_transforms import Query_Resolver_Transform, Document_Resolver_Transform, \
                                                      Query_Doc_Merge_Transform, BERT_Numericalise_Transform, \
                                                      q_id_Numericalize_Transform, d_id_Numericalize_Transform
from src.useful_utils import chunks

from tqdm.auto import tqdm 
import torch
import random

class Manual_Query_Doc_Pipe_Transform():
    def __init__(self, get_query_fn, get_doc_fn):
        
        self.transforms_pipe = [
            Query_Resolver_Transform(get_query_fn, utterance_type="manual_rewritten_utterance"),
            Document_Resolver_Transform(get_doc_fn),
            Query_Doc_Merge_Transform(),
            BERT_Numericalise_Transform(),
            q_id_Numericalize_Transform(),
            d_id_Numericalize_Transform()
        ]
    def __call__(self, samples):
        '''
        samples: dict: {'q_id':"31_4", 'd_id':"CAR_xxx", ...}
        returns: dict: {'input_ids':[34,2,8...], 'q_id_ascii':[55,41,...]}
        '''
        for transform in self.transforms_pipe:
            samples = transform(samples)
        return samples
    
class RUN_File_Search_Transform():
    def __init__(self, run_file, hits=100, **kwargs):
        '''
        first_pass_model_fn: (q_id) -> [(d_id, score), ...]
        '''
        self.first_pass_model_fn = Run_File_Searcher(run_file, **kwargs).predict
        self.hits = hits
        
    def __call__(self, samples):
        '''
        samples: [dict]: [{'q_id':"xxx", ...}]
        returns: [dict]: [{'q_id':"xxx", 'search_results':[("MARCO_xxx", 0.4), ("CAR_xxx",0.3)..], ...}]
        '''
        for sample_obj in tqdm(samples, desc="Searching queries"):
            q_id = sample_obj["q_id"]
            results = self.first_pass_model_fn(q_id, hits=self.hits)
            sample_obj["search_results"] = results
        return samples
    
class BM25_Search_Transform():
    def __init__(self, hits=100, **kwargs):
        '''
        first_pass_model_fn: ("query text") -> [(d_id, score), ...]
        '''
        self.first_pass_model_fn = BM25_Ranker(**kwargs).predict
        self.hits = hits
        
    def __call__(self, samples):
        '''
        samples: [dict]: [{'query':"query text", ...}]
        returns: [dict]: [{'query':"query text", 'search_results':[("MARCO_xxx", 0.4), ("CAR_xxx",0.3)..], ...}]
        '''
        for sample_obj in tqdm(samples, desc="Searching queries"):
            query = sample_obj["query"]
            results = self.first_pass_model_fn(query, hits=self.hits)
            sample_obj["search_results"] = results
        return samples
    
class BERT_Score_Transform():
    def __init__(self, checkpoint_path, batch_size=64, PAD_id = 0, **kwargs):
        '''
        checkpoint_path: str: path to only the state dict of the model, loaded with load_state_dict
        '''
        self.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
        
        self.BERT_Reranker = BERT_Reranker().to(self.device)
        print(self.BERT_Reranker.load_state_dict(torch.load(checkpoint_path)))
        self.BERT_Reranker.eval()
        self.batch_size = batch_size
        self.PAD = PAD_id
        
        print(f"BERT ReRanker initialised on {self.device}. Batch size {batch_size}")
        
    def __call__(self, samples):
        '''
        samples: [dict]: [{'input_ids':[34,2,8...]}]
        returns: [dict]: [{'input_ids':[34,2,8...], "score":0.56}]
        '''
        for sample_obj_batch in chunks(samples, self.batch_size):
            with torch.no_grad():
                input_tensor = torch.nn.utils.rnn.pad_sequence(
                                [torch.tensor(sample_obj["input_ids"], dtype=torch.long) for sample_obj in sample_obj_batch], 
                                                     padding_value=self.PAD).T.to(self.device)
                attention_mask = (input_tensor != self.PAD).type(torch.float).to(self.device)
                scores = self.BERT_Reranker(input_tensor, attention_mask=attention_mask).view(-1).tolist()
            for sample_obj, score in zip(sample_obj_batch, scores):
                sample_obj["score"] = score
                
        return samples
    
class BERT_ReRanker_Transform():
    def __init__(self, checkpoint_path, get_doc_fn, **kwargs):
        '''
        A Transform that reorders a list based on BERT query doc score
        
        checkpoint_path: str: path to only the state dict of the model, loaded with load_state_dict
        '''
        self.BERT_score_transform = BERT_Score_Transform(checkpoint_path, **kwargs)
        self.doc_resolver_transform = Document_Resolver_Transform(get_doc_fn)
        self.q_d_merge_transform = Query_Doc_Merge_Transform()
        self.BERT_numericalize_transform = BERT_Numericalise_Transform()
        
    def __call__(self, samples):
        '''
        samples: [dict]: [{'query':"query text",'search_results':[("MARCO_xxx", 0.4), ("CAR_xxx",0.3)..]...}]
        returns: [dict]: [{'query':"query text",'reranked_results':[("CAR_xxx", 0.54), ("CAR_xxx",0.27)..]...}]
        '''
        for sample_obj in tqdm(samples, desc="Reranking queries"):
            query = sample_obj["query"]
            reranking_samples = [{'query':query, 'd_id':d_id} for d_id, score in sample_obj["search_results"]]
            reranking_samples = self.doc_resolver_transform(reranking_samples)
            reranking_samples = self.q_d_merge_transform(reranking_samples)
            reranking_samples = self.BERT_numericalize_transform(reranking_samples)
            reranking_samples = self.BERT_score_transform(reranking_samples)
            ordered_samples = sorted(reranking_samples, key=lambda sample: sample['score'], reverse=True)
            sample_obj["reranked_results"] = [(sample['d_id'], sample['score']) for sample in ordered_samples]
        return samples
    
class Oracle_ReRanker_Transform():
    def __init__(self, q_rels):
        '''
        A Transform that reorders a list of documents according to the q_rels specified
        q_rels: dict: {'q_id':"32_4", 'q_rel':["CAR_xxx",..]}
        '''
        self.q_rels = q_rels
        
    def __call__(self, samples):
        '''
        samples: [dict]: [{'q_id':'xxx', 'search_results':[("MARCO_xxx", 0.4), ("CAR_xxx",0.3)..],...}]
        returns: [dict]: [{'q_id':'xxx', 'reranked_results':[("CAR_xxx", 0.54), ("CAR_xxx",0.27)..]...}]
        '''
        for sample_obj in samples:
            q_rel = self.q_rels[sample_obj["q_id"]] # ["CAR_xxx",..]
            scored_d_ids = [(d_id, 1 if d_id in q_rel else 0) for d_id, score in sample_obj["search_results"]]
            sample_obj["reranked_results"] = sorted(scored_d_ids, key=lambda res: res[0], reverse=True)
        return samples
    
class Random_ReRanker_Transform():
    def __init__(self):
        '''
        A Transform that reorders a list of documents randomly for refrence
        '''
        
    def __call__(self, samples):
        '''
        samples: [dict]: [{'q_id':'xxx', 'search_results':[("MARCO_xxx", 0.4), ("CAR_xxx",0.3)..],...}]
        returns: [dict]: [{'q_id':'xxx', 'reranked_results':[("CAR_xxx", 0.54), ("CAR_xxx",0.27)..]...}]
        '''
        for sample_obj in samples:
            sample_obj["reranked_results"] = random.shuffle(sample_obj["search_results"][:])
        return samples