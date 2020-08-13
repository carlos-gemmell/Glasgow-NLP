import torch
from torch.nn import functional as F
from torch import nn
import random
from pytorch_lightning.core.lightning import LightningModule
from transformers import BartModel, BertTokenizer, BartForConditionalGeneration

from src.RawDataLoaders import MS_Marco_RawDataLoader
from src.pipe_datasets import Manual_Query_BM25_Reranking_Dataset
from src.models_and_transforms.text_transforms import Reranking_Sampler_Transform, q_id_Denumericalize_Transform, d_id_Denumericalize_Transform, \
                                                      BART_Denumericalise_Transform
from src.models_and_transforms.complex_transforms import Manual_Query_Doc_Pipe_Transform
from src.Experiments import Sequence_Similarity_Experiment

class BART_Query_ReWriter(LightningModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.BART = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

        
    def forward(self, encoder_input, decoder_input):
        outputs = self.BART(input_ids, decoder_input_ids=decoder_input)
        return outputs
        
    def training_step(self, batch, batch_idx):
        encoder_input = batch["input_ids"]
        input_mask = batch['input_attention_mask']
        
        decoder_input = batch['decoder_input_ids']
        decoder_target = batch['decoder_target_ids']
        decoder_mask = batch['target_attention_mask']
                
        outputs = self.BART(encoder_input, 
                            decoder_input_ids=decoder_input, 
                            attention_mask=input_mask, 
                            decoder_attention_mask=decoder_mask, 
                            use_cache=False)  
        logits = outputs[0]
                
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.BART.config.vocab_size), decoder_target.view(-1))
        if torch.isnan(loss):
            print(f'input_ids is nan:{torch.isnan(batch["input_ids"])}, decoder_input_ids is nan:{torch.isnan(batch["decoder_input_ids"])}')
            print(f'logits={logits}')
            
        return {"loss":loss, 'logits':logits}
    
    def validation_step(self, batch, batch_idx):
        encoder_input = batch["input_ids"]
        input_mask = batch['input_attention_mask']
        decoder_target = batch['decoder_target_ids']
        
        outputs = self.generate(encoder_input, attention_mask=input_mask, num_beams=4, max_length=40, early_stopping=True)
        batch["generated_ids"] = outputs
        return batch
        
        
    def validation_epoch_end(self, outputs):
        samples = []
        for batch in outputs:
            for i in range(len(batch["input_ids"])):
                sample_obj = {}
                sample_obj["generated_ids"] = batch["generated_ids"][i].tolist()
                sample_obj["input_ids"] = batch["input_ids"][i].tolist()
                sample_obj["target_ids"] = batch['decoder_target_ids'][i].tolist()
                samples.append(sample_obj)
            
        denumericalize_transform = BART_Denumericalise_Transform(fields=[('generated_ids','predicted_seq'),
                                                                       ('input_ids', 'input_text'),
                                                                       ('target_ids', 'target_seq')])
        samples = denumericalize_transform(samples)
        experiment = Sequence_Similarity_Experiment()
        metrics = experiment(samples)
        print(metrics)
        random_sample = random.choice(samples)
        print("-----EXAMPLE------")
        print(f"Input Text: '{random_sample['input_text']}'")
        print(f"      Pred: '{random_sample['predicted_seq']}'")
        print(f"    Target: '{random_sample['target_seq']}'")
        print("------------------")
        return {'val_loss':metrics["BLEU"], 'log':metrics}
            
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.000001)
    
    def backward(self, use_amp, loss, optimizer, _):
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        
    def generate(self, *args,**kwargs):
        return self.BART.generate(*args,**kwargs)