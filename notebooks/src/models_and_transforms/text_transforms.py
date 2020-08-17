import copy
from transformers import BertTokenizer, BartTokenizer
from tqdm.auto import tqdm 
import random
import ujson

class Reranking_Flattener_Transform():
    def __init__(self, **kwargs):
        '''
        A Transform that flattens the search results into a series of samples.
        '''
        pass
    
    def __call__(self, samples):
        '''
        sample_obj: [dict]: [{'q_id':"xxx", 'search_results':[("MARCO_xxx", 0.4), ("CAR_xxx",0.3)..], ...}]
        returns: [dict]: [{'q_id':"xxx", 'd_id':"CAR_xxx", ...}]
        '''
        new_samples = []
        for sample_obj in tqdm(samples, desc='Flattening search results'):
            results = sample_obj["search_results"]
            for d_id, score in results:
                new_sample = copy.deepcopy(sample_obj)
                new_sample["d_id"] = d_id
                new_sample.pop('search_results', None)
                new_samples.append(new_sample)
        return new_samples

class Reranking_Sampler_Transform():
    def __init__(self, num_neg_samples=100000,**kwargs):
        self.num_neg_samples = num_neg_samples
    
    def __call__(self, samples):
        '''
        sample_obj: [dict]: [{'q_rel':["MARCO_xxx"], 'search_results':[("MARCO_xxx", 0.4), ("CAR_xxx",0.3)..], ...}]
        returns: [dict]: [{'d_id':"CAR_xxx", 'label':0/1, 'q_rel':["MARCO_xxx"], ...}]
        '''
        new_samples = []
        for sample_obj in tqdm(samples, desc="Sampling ± query-doc pairs"):
            q_rels = sample_obj["q_rel"]
            results = sample_obj["search_results"]
            random.shuffle(results)
            for d_id, score in results[:self.num_neg_samples]:
                if d_id in q_rels:
                    continue
                neg_sample = ujson.loads(ujson.dumps(sample_obj))#copy.deepcopy(sample_obj)
                neg_sample["d_id"] = d_id
                neg_sample["label"] = 0
                neg_sample.pop('search_results', None) # remove because it is un-necessarily big to store
                new_samples.append(neg_sample)
                
                pos_sample = ujson.loads(ujson.dumps(sample_obj))#copy.deepcopy(sample_obj)
                pos_sample["d_id"] = random.choice(q_rels)
                pos_sample["label"] = 1
                pos_sample.pop('search_results', None) # remove because it is un-necessarily big to store
                new_samples.append(pos_sample)
        return new_samples
                
class Real_Time_Reranking_Sampler_Transform():
    def __init__(self, first_pass_model_fn, hits=100):
        '''
        first_pass_model_fn: (q_id) -> [(d_id, score), ...]
        '''
        self.first_pass_model_fn = first_pass_model_fn
        self.hits = hits
    
    def __call__(self, samples):
        '''
        sample_obj: [dict]: [{'q_id':"31_4", 'q_rel':["MARCO_xxx"], ...}]
        returns: [dict]: [{'d_id':"CAR_xxx", 'label':0/1, 'q_id':"31_4", 'q_rel':["MARCO_xxx"], ...}]
        '''
        for sample_obj in samples:
            q_id = sample_obj["q_id"]
            q_rels = sample_obj["q_rel"]
            true_label = int(random.random() > 0.5)
            d_id = self.sample_positive(q_rels) if true_label else self.sample_negative(q_id, q_rels)
            sample_obj["d_id"] = d_id
            sample_obj["label"] = true_label
        return samples
            
    def sample_positive(self, q_rels):
        return random.choice(q_rels)
    
    def sample_negative(self, q_id, q_rels):
        results = self.first_pass_model_fn(q_id, hits=self.hits)
        for i in range(100):
            d_id, score =  random.choice(results)
            if d_id not in q_rels:
                return d_id
        print("Sampled too many times, no negative found.")
        
class Query_Resolver_Transform():
    def __init__(self, get_query_fn, utterance_type="manual_rewritten_utterance", **kwargs):
        '''
        get_query_fn: fn(q_id) -> "query string"
        utterance_type: str: "manual_rewritten_utterance", "automatic_rewritten_utterance", "raw_utterance"
        '''
        self.get_query_fn = get_query_fn
        self.utterance_type = utterance_type
    
    def __call__(self, samples):
        '''
        sample_obj: [dict]: [{'q_id':"31_4", ...}]
        returns: [dict]: [{'query':"query text", 'q_id':"31_4", ...}]
        '''
        for sample_obj in samples:
            sample_obj["query"] = self.get_query_fn(sample_obj["q_id"], utterance_type=self.utterance_type)
        return samples
            
class Document_Resolver_Transform():
    def __init__(self, get_doc_fn, **kwargs):
        '''
        get_doc_fn: fn(d_id) -> "document string"
        '''
        self.get_doc_fn = get_doc_fn
    
    def __call__(self, samples):
        '''
        sample_obj: [dict]: [{'d_id':"CAR_xxx", ...}]
        returns: [dict]: [{'doc':"document text", 'd_id':"CAR_xxx", ...}]
        '''
        for sample_obj in samples:
            sample_obj["doc"] = self.get_doc_fn(sample_obj["d_id"])
        return samples
        
class Query_Doc_Merge_Transform():
    def __init__(self, separator=" [SEP] ", **kawrgs):
        self.separator = separator
    
    def __call__(self, samples):
        '''
        sample_obj: [dict]: [{'query':"query text", 'doc':"doc text", ...}]
        returns: [dict]: [{'input_text':"q text [SEP] d text", 'query':"query text", 'doc':"doc text", ...}]
        '''
        for sample_obj in samples:
            sample_obj["input_text"] = sample_obj["query"] + " [SEP] " + sample_obj["doc"]
        return samples

class BERT_Numericalise_Transform():
    def __init__(self):
        self.numericalizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    def __call__(self, samples):
        '''
        sample_obj: [dict]: [{'input_text':"text and more", ...}]
        returns: [dict]: [{'input_ids':[34,2,8...], 'input_text':"text and more", ...}]
        '''
        for sample_obj in samples:
            sample_obj["input_ids"] = self.numericalizer.encode(sample_obj["input_text"], truncation=True)
        return samples
    
class MonoBERT_Numericalise_Transform():
    def __init__(self, vocab_txt_file="saved_models/monoBERT/vocab.txt", **kwargs):
        self.numericalizer = BertTokenizer(vocab_txt_file)
    
    def __call__(self, samples):
        '''
        sample_obj: [dict]: [{'query':"text and more", 'doc':"doc text" ...}]
        returns: [dict]: [{'input_ids':[34,2,8...], 'type_ids':[0,0,1,1], 'input_text':"text and more", ...}]
        '''
        for sample_obj in samples:
            query_text = sample_obj['query']
            query_ids = [self.numericalizer.cls_token_id] + self.numericalizer.encode(query_text, add_special_tokens=False)[:62] + [self.numericalizer.sep_token_id]
            query_token_type_ids = [0]*len(query_ids)
            
            doc_text = sample_obj['doc']
            doc_ids = self.numericalizer.encode(doc_text, add_special_tokens=False)[:445] + [self.numericalizer.sep_token_id]
            doc_token_type_ids = [1]*len(doc_ids)
            
            sample_obj["input_ids"] = query_ids+doc_ids
            sample_obj["type_ids"] = query_token_type_ids+doc_token_type_ids
        return samples
    
class DuoBERT_Numericalise_Transform():
    def __init__(self, vocab_txt_file=None):
        self.numericalizer = BertTokenizer(vocab_txt_file)
    
    def __call__(self, samples):
        '''
        sample_obj: [dict]: [{'query':"text and more", 'docA':"docA text", 'docB':"docB text" ...}]
        returns: [dict]: [{'input_ids':[34,2,8...], 'type_ids':[0,0,1,1], 'input_text':"text and more", ...}]
        '''
        for sample_obj in samples:
            query_text = sample_obj['query']
            query_ids = [self.numericalizer.cls_token_id]+self.numericalizer.encode(query_text, add_special_tokens=False)[:62]+[self.numericalizer.sep_token_id]
            query_token_type_ids = [0]*len(query_ids)
            
            docA_text = sample_obj['docA']
            docA_ids = self.numericalizer.encode(docA_text, add_special_tokens=False)[:223] + [self.numericalizer.sep_token_id]
            docA_token_type_ids = [1]*len(docA_ids)
            
            docB_text = sample_obj['docB']
            docB_ids = self.numericalizer.encode(docB_text, add_special_tokens=False)[:223] + [self.numericalizer.sep_token_id]
            docB_token_type_ids = [2]*len(docB_ids)
            
            sample_obj["input_ids"] = query_ids+docA_ids+docB_ids
            sample_obj["type_ids"] = query_token_type_ids+docA_token_type_ids+docB_token_type_ids
        return samples
    
class BART_Numericalise_Transform():
    def __init__(self, fields=[("input_text","input_ids")]):
        self.numericalizer = BartTokenizer.from_pretrained('facebook/bart-large')
        self.fields = fields
    
    def __call__(self, samples):
        '''
        sample_obj: [dict]: [{'input_text':"text and more", ...}]
        returns: [dict]: [{'input_ids':[34,2,8...], 'input_text':"text and more", ...}]
        '''
        for sample_obj in samples:
            for str_field, id_field in self.fields:
                sample_obj[id_field] = self.numericalizer.encode(sample_obj[str_field])
        return samples

class BERT_Denumericalise_Transform():
    def __init__(self, fields=[("input_ids","input_text")]):
        self.numericalizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.fields = fields
    
    def __call__(self, samples):
        '''
        sample_obj: [dict]: [{'input_ids':[34,2,8...], ...}]
        returns: [dict]: [{'input_ids':[34,2,8...], 'input_text':"text and more", ...}]
        '''
        for sample_obj in samples:
            for str_field, id_field in self.fields:
                sample_obj[id_field] = self.numericalizer.decode(sample_obj[str_field])
        return samples

class BART_Denumericalise_Transform():
    def __init__(self, fields=[("input_ids","input_text")]):
        self.numericalizer = BartTokenizer.from_pretrained('facebook/bart-large')
        self.fields = fields
    
    def __call__(self, samples):
        '''
        sample_obj: [dict]: [{'input_ids':[34,2,8...], ...}]
        returns: [dict]: [{'input_ids':[34,2,8...], 'input_text':"text and more", ...}]
        '''
        for sample_obj in samples:
            for str_field, id_field in self.fields:
                sample_obj[id_field] = self.numericalizer.decode(sample_obj[str_field],skip_special_tokens=True)
        return samples
    
class q_id_Numericalize_Transform():
    def __init__(self, pad_size=64):
        '''
        Creates a numericall version of the q_id passed that adheres to the pad_size so it can be converttetd to a tensor.
        '''
        self.pad_size = pad_size
    
    def __call__(self, samples):
        '''
        sample_obj" [dict]: [{'q_id':"MARCO_0",...}]
        returns: [dict]: [{'q_id':"MARCO_0", 'q_id_ascii':[55,41,...],...}]
        '''
        for sample_obj in samples:
            sample_obj['q_id_ascii'] = [ord(c) for c in sample_obj['q_id']]
            sample_obj['q_id_ascii'] += [-1]*(self.pad_size-len(sample_obj['q_id_ascii']))
        return samples
            

class q_id_Denumericalize_Transform():
    def __init__(self):
        pass
    
    def __call__(self, samples):
        '''
        sample_obj" [dict]: [{'q_id_ascii':[55,41,...],...}]
        returns: [dict]: [{'q_id':"MARCO_0", 'q_id_ascii':[55,41,...],...}]
        '''
        for sample_obj in samples:
            sample_obj['q_id'] = ''.join([chr(ascii_val) for ascii_val in sample_obj['q_id_ascii'] if ascii_val != -1])
        return samples
    
class d_id_Numericalize_Transform():
    def __init__(self, pad_size=64):
        '''
        Creates a numericall version of the d_id passed that adheres to the pad_size so it can be converttetd to a tensor.
        '''
        self.pad_size = pad_size
    
    def __call__(self, samples):
        '''
        sample_obj" [dict]: [{'d_id':"MARCO_0",...}]
        returns: [dict]: [{'d_id':"MARCO_0", 'd_id_ascii':[55,41,...],...}]
        '''
        for sample_obj in samples:
            sample_obj['d_id_ascii'] = [ord(c) for c in sample_obj['d_id']]
            sample_obj['d_id_ascii'] += [-1]*(self.pad_size-len(sample_obj['d_id_ascii']))
        return samples
            

class d_id_Denumericalize_Transform():
    def __init__(self):
        pass
    
    def __call__(self, samples):
        '''
        sample_obj" [dict]: [{'d_id_ascii':[55,41,...],...}]
        returns: [dict]: [{'d_id':"MARCO_0", 'd_id_ascii':[55,41,...],...}]
        '''
        for sample_obj in samples:
            sample_obj['d_id'] = ''.join([chr(ascii_val) for ascii_val in sample_obj['d_id_ascii'] if ascii_val != -1])
        return samples
    
class Rewriter_Query_Resolver_Transform():
    def __init__(self, get_query_fn, **kwargs):
        '''
        get_query_fn: fn(q_id) -> "query string"
        '''
        self.get_query_fn = get_query_fn
        
    def __call__(self, samples):
        '''
        samples [dict]: [{'q_id':"32_4", 'prev_turns':["32_3",..]},...]
        returns: [dict]: [{'unresolved_query':'query text', 'resolved_query':'query text', 'previous_queries':['first query text', 'second query text']}]
        '''
        for sample_obj in samples:
            sample_obj["unresolved_query"] = self.get_query_fn(sample_obj["q_id"], utterance_type='raw_utterance')
            resolved_previous_queries = [self.get_query_fn(q_id, utterance_type='manual_rewritten_utterance')for q_id in sample_obj["prev_turns"]]
            sample_obj["previous_queries"] = resolved_previous_queries
            sample_obj["resolved_query"] = self.get_query_fn(sample_obj["q_id"], utterance_type='manual_rewritten_utterance')
        return samples
    
class Rewriter_Context_Query_Merge_Transform():
    def __init__(self, **kwargs):
        '''
        This Transform merges queries from previous turns and the current unresolved query into a single input sequence.
        '''
        pass
    
    def __call__(self, samples):
        '''
        samples: [dict]: [{'unresolved_query':'query text', 'previous_queries':['first query text', 'second query text']}]
        returns: [dict]: [{'input_text':'merged query text', 'unresolved_query':'query text', 'previous_queries':['first query text',]}]
        '''
        for sample_obj in samples:
            sample_obj["input_text"] = " ".join(sample_obj['previous_queries']) + " query: " + sample_obj['unresolved_query']
        return samples
    
class Rewriter_Context_Target_Transform():
    def __init__(self,  merge_mode="full_context_rewrite", **kwargs):
        '''
        This Transform merges queries from previous turns and the current RESOLVED target query to make the target sequence to be predicted.
        '''
        self.merge_mode = merge_mode
    
    def __call__(self, samples):
        '''
        samples: [dict]: [{'resolved_query':'resolved query text', 'previous_queries':['first query text', 'second query text']}]
        returns: [dict]: [{'target_text':'merged query text', 'unresolved_query':'query text', 'previous_queries':['first query text',]}]
        '''
        for sample_obj in samples:
            if self.merge_mode == "full_context_rewrite":
                sample_obj["target_text"] = " ".join(sample_obj['previous_queries']) + " query: " + sample_obj['resolved_query']
            elif self.merge_mode == "last_turn_rewrite":
                sample_obj["target_text"] = sample_obj['resolved_query']
        return samples
    
class Query_Cleaner_Transform():
    def __init__(self, fields=[('query','cleaned_query')]):
        '''
        This Transform removes some of  the un-necessary halucinated text from the query re-writer.
        '''
        self.fields = fields
        
    def __call__(self, samples):
        '''
        samples: [dict]: [{'query':"query: query text? what else?!"}]
        returns: [dict]: [{'cleaned_query':"query text?", 'query':"query: query text? what else?!"}]
        '''
        for sample_obj in samples:
            for input_field, target_field in self.fields:
                query = sample_obj[input_field]
                new_query = query.split("query:")[-1]
                if "?" in new_query:
                    new_query = new_query[:new_query.index('?')+1]
                sample_obj[target_field] = new_query
        return samples