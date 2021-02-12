import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from transformers import LongformerForSequenceClassification, LongformerModel, BertModel, BertTokenizer, BertForSequenceClassification, \
                         get_linear_schedule_with_warmup, BertConfig
import numpy as np
from transformers.optimization import AdamW

from src.RawDataLoaders import MS_Marco_RawDataLoader
# from src.pipe_datasets import Manual_Query_BM25_Reranking_Dataset
from src.models_and_transforms.text_transforms import Reranking_Sampler_Transform, q_id_Denumericalize_Transform, d_id_Denumericalize_Transform
# from src.models_and_transforms.complex_transforms import Manual_Query_Doc_Pipe_Transform
from src.Experiments import Ranking_Experiment

from stable_baselines3.common.policies import ActorCriticPolicy

import collections
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from stable_baselines3.common.distributions import (
    CategoricalDistribution,
    make_proba_distribution,
)
from stable_baselines3.common.preprocessing import get_action_dim, is_image_space, preprocess_obs
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor, MlpExtractor, NatureCNN, create_mlp
from stable_baselines3.common.utils import get_device, is_vectorized_observation
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.vec_env.obs_dict_wrapper import ObsDictWrapper


class CausalBERT(LightningModule):
    def __init__(self, config, pad_id, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.pad_id = pad_id
        self.transformer = BertModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.value_head = nn.Linear(config.hidden_size, 1, bias=False)
    
    def forward(self, input_ids, **kwargs):
        batch_size = input_ids.shape[0]
        amount_pad = (input_ids == self.pad_id).sum(dim=1)
        
        attention_mask = (input_ids != self.pad_id)
        position_ids = torch.cumsum(attention_mask, dim=1)
        
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids)
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)
        value_logits = self.value_head(hidden_states)
        values = torch.tanh(value_logits)
        return lm_logits, values
    
    def predict(self, x, **kwargs):
        x = x.to(self.device)
        policy_logits, value_logits = self.forward(x, **kwargs)
        policy_logit, value_logit = policy_logits[:,-1], value_logits[:, -1]
        p = torch.softmax(policy_logit, dim=-1)
        return p.data.cpu(),  value_logit.data.cpu()
        
    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        target_policies = batch["target_policies"]
        target_values = batch['target_values']
        grad_mask = batch['not_auto_gen_mask']
        batch_size = input_ids.shape[0]
        
        policy_logits, value_logits = self(input_ids)
        
        value_loss = nn.MSELoss()(value_logits[grad_mask].view(-1), target_values[grad_mask].view(-1))
        
        policy_loss = -torch.dot(target_policies[grad_mask].view(-1), torch.log(policy_logits[grad_mask].softmax(-1)).view(-1))/grad_mask.sum()
        
        if torch.isnan(policy_loss):
            print(-torch.dot(target_policies[grad_mask].view(-1), torch.log(policy_logits[grad_mask].softmax(-1)).view(-1))/grad_mask.sum())
            
        loss = policy_loss + value_loss
            
        return {"loss":loss, 'log': {'train_loss': loss}}
    
    def configure_optimizers(self):
        self.lr=0.0001
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
#         scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=10000, epochs=1)
#         scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=15000)
        return optimizer#, [scheduler]
    
        
            


class BERT_Reranker(LightningModule):
    def __init__(self, h_dim=768, **kwargs):
        super().__init__(**kwargs)
#         self.data_processor = data_processor
        self.BERT_for_class = BertModel.from_pretrained('bert-base-uncased')
        self.testing = False
        self.training = True
        self.dropout = nn.Dropout(0.5)
        self.proj_layer = nn.Linear(h_dim, 1)

        
    def forward(self, x, **kwargs):
        outputs = self.BERT_for_class(x, **kwargs)
        pooled_out = outputs[1]
        pooled_out = self.dropout(pooled_out)
        logit = self.proj_layer(pooled_out)
        return logit
        
    def training_step(self, batch, batch_idx):
        encoder_input = batch["input_ids"]
        labels = batch["label"]
        attention_mask = batch["attention_mask"]
#         loss = self.BERT_for_class(encoder_input, labels=labels, attention_mask=attention_mask)[0]
        output = self(encoder_input, attention_mask=attention_mask)
#         print(output.view(-1), labels.view(-1))
#         loss = nn.BCEWithLogitsLoss()(output.view(-1), labels.view(-1))
        loss = nn.MSELoss()(output.view(-1), labels.view(-1))
            
        return {"loss":loss, 'log': {'train_loss': loss}}
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=0.00001)
#         scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=10000, epochs=1)
#         scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=15000)
        return optimizer#, [scheduler]
    
    
    def validation_step(self, batch, batch_idx):
        encoder_input = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        output = self(encoder_input, attention_mask=attention_mask)
        batch["score"] = output.view(-1)
        return batch
    
    def set_validation_q_rels(self, valid_q_rels):
        '''
        valid_q_rels: dict: {'q_id':[d_id, d_id,...],...}
        '''
        self.valid_q_rels = valid_q_rels
    
    def validation_epoch_end(self, outputs):
        assert hasattr(self, 'valid_q_rels'), "Cannot find validation q_rels, please provide using model.set_validation_q_rels(valid_q_rels)"
        
        samples = []
        for output in outputs:
            batch_size = len(output[list(output.keys())[0]])
            for i in range(batch_size):
                new_sample = {}
                for key in output.keys():
                    new_sample[key] = output[key][i].tolist()
                samples.append(new_sample)
        
        samples = q_id_Denumericalize_Transform()(samples)
        samples = d_id_Denumericalize_Transform()(samples)
#         for sample in samples:
#             print(f"q_id: {sample['q_id']}; d_id: {sample['d_id']}; IDs: {sample['input_ids']}; Score: {sample['score']}")
        search_res_samples_dict = {}
        for sample in samples:
            q_id = sample['q_id']
            if q_id not in search_res_samples_dict:
                search_res_samples_dict[q_id] = []
            
            search_res_samples_dict[q_id].append((sample['d_id'], sample['score']))
            
        search_res_samples = [{'q_id':q_id, 'search_results':search_results} for q_id, search_results in search_res_samples_dict.items()]
            
#         print(search_res_samples[0]['q_id'])
#         print(f"Model training: {self.training}")
#         print(search_res_samples[0]['search_results'])
        
        experiment = Ranking_Experiment(self.valid_q_rels)
        metrics = experiment(search_res_samples)
        
        print("saving model")
        torch.save(self.state_dict(), f"saved_models/BERT_reranker_q500k_h150_checkpoints/BERT_ReRanker_MARCO_from_valid_{metrics['ndcg']}.ckpt")
        
        print(f"Total val samples: {len(search_res_samples)}, number of queries: {len(list(search_res_samples_dict.keys()))}")
        
        print(metrics)
        return {'log':metrics, 'ndcg':torch.tensor(metrics['ndcg']), 'set_recall':torch.tensor(metrics['set_recall'])}
    
    def backward(self, use_amp, loss, optimizer, _):
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        
        
class BertForPassageRanking(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.weight = torch.autograd.Variable(torch.ones(2, config.hidden_size),
                                              requires_grad=True)
        self.bias = torch.autograd.Variable(torch.ones(2), requires_grad=True)