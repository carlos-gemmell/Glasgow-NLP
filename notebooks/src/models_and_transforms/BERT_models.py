import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from transformers import LongformerForSequenceClassification, LongformerModel, BertModel, BertTokenizer, BertForSequenceClassification, \
                         get_linear_schedule_with_warmup
from transformers.optimization import AdamW

from src.RawDataLoaders import MS_Marco_RawDataLoader
# from src.pipe_datasets import Manual_Query_BM25_Reranking_Dataset
from src.models_and_transforms.text_transforms import Reranking_Sampler_Transform, q_id_Denumericalize_Transform, d_id_Denumericalize_Transform
# from src.models_and_transforms.complex_transforms import Manual_Query_Doc_Pipe_Transform
from src.Experiments import Ranking_Experiment


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