import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from transformers import BartModel, BertTokenizer, BartForConditionalGeneration

from src.RawDataLoaders import MS_Marco_RawDataLoader
from src.pipe_datasets import Manual_Query_BM25_Reranking_Dataset
from src.models_and_transforms.text_transforms import Reranking_Sampler_Transform, q_id_Denumericalize_Transform, d_id_Denumericalize_Transform
from src.models_and_transforms.complex_transforms import Manual_Query_Doc_Pipe_Transform

class BART_Query_ReWriter(LightningModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.BART = BartForConditionalGeneration.from_pretrained('facebook/bart-large')

        
    def forward(self, input_ids):
        logits = self.BART(input_ids)
        return logits
        
    def training_step(self, batch, batch_idx):
        encoder_input = batch["input_ids"]
        input_mask = batch['input_attention_mask']
        
        decoder_input = batch['decoder_input_ids']
        decoder_target = batch['decoder_target_ids']
        labels_mask = batch['target_attention_mask']
        
        outputs = self.BART(encoder_input, decoder_input_ids=decoder_input)  
        logits = outputs[0]
        loss_fct = nn.CrossEntropyLoss()
        print(logits)
        loss = loss_fct(logits.view(-1, self.BART.config.vocab_size), decoder_target.view(-1))
        return {"loss":loss}
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.000001)
    
    def backward(self, use_amp, loss, optimizer, _):
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)