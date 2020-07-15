import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from transformers import LongformerForSequenceClassification

class Longformer_Reranker(LightningModule):
    def __init__(self, data_processor, h_dim=768):
        self.data_processor = data_processor
        self.longformer = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096')
        
    def forward(self, x):
        outputs = self.longformer(x)
        
    def train_step(self, batch):
        total_loss = 0.
        encoder_input = batch["input"].to(self.model.device)
        targets = batch["label"][].to(self.model.device)
        
        self.optimizer.zero_grad()
        logits, pooled_logits = self.model(encoder_input)
        loss_fct = nn.MSELoss()
        loss = loss_fct(logits.view(-1), labels.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()
        return loss