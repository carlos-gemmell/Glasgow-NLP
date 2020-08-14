from src.models_and_transforms.text_transforms import Query_Resolver_Transform, Reranking_Sampler_Transform, Reranking_Flattener_Transform, \
                                                      Rewriter_Query_Resolver_Transform, Rewriter_Context_Query_Merge_Transform, \
                                                      BART_Numericalise_Transform, Rewriter_Context_Target_Transform
from src.models_and_transforms.complex_transforms import BM25_Search_Transform, Manual_Query_Doc_Pipe_Transform, RUN_File_Search_Transform
from src.RawDataLoaders import MS_Marco_RawDataLoader

from transformers import BartTokenizer
from tqdm.auto import tqdm 
from torch.utils.data import Dataset, DataLoader
import torch

class Pipe_Dataset(Dataset):
    def __init__(self, samples, slow_pipe, real_time_pipe, **kwargs):
        
        self.real_time_pipe = real_time_pipe
        self.PAD = 0
        
        pbar = tqdm(slow_pipe)
        for transform in pbar:
            pbar.set_description(transform.__class__.__name__)
            samples = transform(samples)
        self.samples = samples
        
        super().__init__()
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError
        
        sample = self.samples[idx]
        samples = [sample]
        
        for transform in self.real_time_pipe:
            samples = transform(samples)
            assert len(samples) == 1, "The number of samples has increased! Only the first of these will be taken leading to loss of data"
        
        return samples[0]
    
    def collate(self, input_samples):
        """
        input_samples: [dict]: these are samples obtained through the _get_item method
        """
        collated_samples = {}
        sample_keys = input_samples[0].keys()
        for key in sample_keys:
            collated_samples[key] = torch.nn.utils.rnn.pad_sequence([torch.tensor(sample[key], dtype=torch.long) for sample in input_samples], 
                                                 padding_value=self.PAD)
        return collated_samples
    
    def to_dataloader(self, batch_size, num_workers=0, shuffle=False):
        dataloader = DataLoader(self, batch_size=batch_size, num_workers=num_workers,\
                           drop_last=False, collate_fn = self.collate, shuffle=shuffle)
        dataloader.__code__ = 0
        return dataloader
    
class Manual_Query_BM25_Reranking_Dataset(Pipe_Dataset):
    def __init__(self, samples, get_query_fn, get_doc_fn, **kwargs):
        '''
        samples: [dict]: [{'q_id':"32_4", 'q_rel':[MARCO_xxx]}]
        get_doc_fn: fn(d_id) -> "doc string"
        get_query_fn: fn(q_id) -> "query string"
        '''        
        slow_pipe = [Query_Resolver_Transform(get_query_fn),
                     BM25_Search_Transform(**kwargs),
                     Reranking_Sampler_Transform(**kwargs)]
        real_time_pipe = [Manual_Query_Doc_Pipe_Transform(get_query_fn, get_doc_fn)]
        
        super().__init__(samples, slow_pipe, real_time_pipe, **kwargs)
    
    def collate(self, input_samples):
        """
        input_samples: [dict]: these are samples obtained through the __getitem__ method
        """
        collated_samples = {}
        collated_samples["input_ids"] = torch.nn.utils.rnn.pad_sequence([torch.tensor(sample["input_ids"], dtype=torch.long) for sample in input_samples], 
                                                 padding_value=self.PAD).T
        
        collated_samples["attention_mask"] = (collated_samples["input_ids"] != self.PAD).type(torch.float)
        collated_samples["label"] = torch.cat([torch.tensor([sample["label"]], dtype=torch.float) for sample in input_samples])
        collated_samples["q_id_ascii"] = torch.tensor([sample["q_id_ascii"] for sample in input_samples])
        collated_samples["d_id_ascii"] = torch.tensor([sample["d_id_ascii"] for sample in input_samples])
        
        return collated_samples
    
class Manual_Query_RUN_File_Reranking_Dataset(Pipe_Dataset):
    def __init__(self, samples, get_query_fn, get_doc_fn, run_file, **kwargs):
        '''
        samples: [dict]: [{'q_id':"32_4", 'q_rel':[MARCO_xxx]}]
        get_doc_fn: fn(d_id) -> "doc string"
        get_query_fn: fn(q_id) -> "query string"
        '''        
        slow_pipe = [Query_Resolver_Transform(get_query_fn),
                     RUN_File_Search_Transform(run_file, **kwargs),
                     Reranking_Sampler_Transform(**kwargs)]
        real_time_pipe = [Manual_Query_Doc_Pipe_Transform(get_query_fn, get_doc_fn)]
        
        super().__init__(samples, slow_pipe, real_time_pipe, **kwargs)
    
    def collate(self, input_samples):
        """
        input_samples: [dict]: these are samples obtained through the __getitem__ method
        """
        collated_samples = {}
        collated_samples["input_ids"] = torch.nn.utils.rnn.pad_sequence([torch.tensor(sample["input_ids"], dtype=torch.long) for sample in input_samples], 
                                                 padding_value=self.PAD).T
        
        collated_samples["attention_mask"] = (collated_samples["input_ids"] != self.PAD).type(torch.float)
        collated_samples["label"] = torch.cat([torch.tensor([sample["label"]], dtype=torch.float) for sample in input_samples])
        collated_samples["q_id_ascii"] = torch.tensor([sample["q_id_ascii"] for sample in input_samples])
        collated_samples["d_id_ascii"] = torch.tensor([sample["d_id_ascii"] for sample in input_samples])
        
        return collated_samples
    
class Reranking_Validation_Dataset(Pipe_Dataset):
    def __init__(self, samples, get_query_fn, get_doc_fn, **kwargs):
        '''
        samples: [dict]: [{'q_id':"32_4", 'search_results':[("MARCO_xxx", 0.4), ("CAR_xxx",0.3)..]}]
        get_doc_fn: fn(d_id) -> "doc string"
        get_query_fn: fn(q_id) -> "query string"
        '''        
        slow_pipe = [Reranking_Flattener_Transform()]
        real_time_pipe = [Manual_Query_Doc_Pipe_Transform(get_query_fn, get_doc_fn)]
        
        super().__init__(samples, slow_pipe, real_time_pipe, **kwargs)
    
    def collate(self, input_samples):
        """
        input_samples: [dict]: these are samples obtained through the __getitem__ method
        """
        collated_samples = {}
        collated_samples["input_ids"] = torch.nn.utils.rnn.pad_sequence([torch.tensor(sample["input_ids"], dtype=torch.long) for sample in input_samples], 
                                                 padding_value=self.PAD).T
        
        collated_samples["attention_mask"] = (collated_samples["input_ids"] != self.PAD).type(torch.float)
        collated_samples["q_id_ascii"] = torch.tensor([sample["q_id_ascii"] for sample in input_samples])
        collated_samples["d_id_ascii"] = torch.tensor([sample["d_id_ascii"] for sample in input_samples])
        
        return collated_samples
    
    
class MARCO_Manual_Query_BM25_Reranking_Dataset(Manual_Query_BM25_Reranking_Dataset):
    def __init__(self, num_queries, neg_hit_depth, neg_hit_samples, **kwargs):
        raw_data_loader = MS_Marco_RawDataLoader(from_pickle=True)
        get_query_fn = raw_data_loader.get_query
        get_doc_fn = raw_data_loader.get_doc
        
        train_raw_samples = raw_data_loader.get_topics("train")
        super().__init__(train_raw_samples[:num_queries], 
                         get_query_fn, 
                         get_doc_fn, 
                         hits=neg_hit_depth, 
                         num_neg_samples=neg_hit_samples, 
                         index_dir="datasets/MS_MARCO/MARCO_anserini")
        
        
class CAsT_Query_ReWriting_Dataset(Pipe_Dataset):
    def __init__(self, samples, get_query_fn, **kwargs):
        '''
        samples [dict]: [{'q_id':"32_4", 'prev_turns':["32_3",..]},...]
        '''
        slow_pipe = [Rewriter_Query_Resolver_Transform(get_query_fn)]
        fast_pipe = [
            Rewriter_Context_Query_Merge_Transform(**kwargs),
            Rewriter_Context_Target_Transform(**kwargs),
            BART_Numericalise_Transform(fields=[('input_text', 'input_ids'), ('target_text', 'target_ids')])
        ]
        
        super().__init__(samples, slow_pipe, fast_pipe, **kwargs)
        self.PAD = BartTokenizer.from_pretrained('facebook/bart-large').pad_token_id
        
    
    def collate(self, input_samples):
        """
        input_samples: [dict]: these are samples obtained through the __getitem__ method
        """
        collated_samples = {}
        collated_samples["input_ids"] = torch.nn.utils.rnn.pad_sequence([torch.tensor(sample["input_ids"], dtype=torch.long) for sample in input_samples], 
                                                 padding_value=self.PAD, batch_first=True)
        
        collated_samples["input_attention_mask"] = (collated_samples["input_ids"] != self.PAD).type(torch.float)
        collated_samples["decoder_input_ids"] = torch.nn.utils.rnn.pad_sequence([torch.tensor(sample["target_ids"][:-1], dtype=torch.long) for sample in input_samples], padding_value=self.PAD, batch_first=True)
        collated_samples["decoder_target_ids"] = torch.nn.utils.rnn.pad_sequence([torch.tensor(sample["target_ids"][1:], dtype=torch.long) for sample in input_samples], padding_value=self.PAD, batch_first=True)
        collated_samples["target_attention_mask"] = (collated_samples["decoder_input_ids"] != self.PAD).type(torch.float)
        
        return collated_samples