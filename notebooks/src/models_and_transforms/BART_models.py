import torch
from torch.nn import functional as F
from torch import nn
import random
from pytorch_lightning.core.lightning import LightningModule
from transformers import BartModel, BertTokenizer, BartForConditionalGeneration

from src.RawDataLoaders import MS_Marco_RawDataLoader
from src.models_and_transforms.text_transforms import Reranking_Sampler_Transform, q_id_Denumericalize_Transform, d_id_Denumericalize_Transform, \
                                                      Denumericalise_Transform
from src.Experiments import Sequence_BLEU_Experiment

class BART_Simple(LightningModule):
    def __init__(self, from_pretrained=True, config=None, **kwargs):
        super().__init__()
        self.lr = 0.0001
        if from_pretrained==False or config:
            self.BART = BartForConditionalGeneration(config)
            # this is to make the random model favour generating the EOS token at the start to not go generating forever
            self.BART.final_logits_bias[0][2]=5.0
        else:
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
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
        return [optimizer], [scheduler]
    
    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure, using_native_amp):
        # warm up lr
        # warm up for 500 steps
#         if self.trainer.global_step < 500:
#             lr_scale = min(1., float(self.trainer.global_step + 1) / 500.)
#             for pg in optimizer.param_groups:
#                 pg['lr'] = lr_scale * self.lr

        # update params
        optimizer.step()
#         optimizer.zero_grad()
    
    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):        
        for param in self.parameters():
            param.grad = None
    
    def backward(self, use_amp, loss, optimizer, _):
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        
    def generate(self, *args,**kwargs):
        return self.BART.generate(*args,**kwargs)
    
    
class BART_Query_ReWriter(BART_Simple):
    def validation_step(self, batch, batch_idx):
        encoder_input = batch["input_ids"]
        input_mask = batch['input_attention_mask']
        decoder_target = batch['decoder_target_ids']
        
        outputs = self.generate(encoder_input, attention_mask=input_mask, num_beams=4, max_length=512, early_stopping=True)
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
            
        denumericalize_transform = Denumericalise_Transform(fields=[('generated_ids','predicted_seq'),
                                                                       ('input_ids', 'input_text'),
                                                                       ('target_ids', 'target_seq')])
        samples = denumericalize_transform(samples)
        for sample_obj in samples:
            sample_obj["target_seq"] = sample_obj["target_seq"].split("query:")[-1]
            query_elements = sample_obj["predicted_seq"].split("query:")
            if len(query_elements) > 1:
                sample_obj["predicted_seq"] = query_elements[-1]
            else:
                sample_obj["predicted_seq"] = query_elements[0]
        experiment = Sequence_BLEU_Experiment()
        metrics = experiment(samples)
        print(metrics)
        for i in range(3):
            random_sample = random.choice(samples)
            print(f"-----EXAMPLE-{i}------")
            print(f"Input Text: '{random_sample['input_text']}'")
            print(f"      Pred: '{random_sample['predicted_seq']}'")
            print(f"    Target: '{random_sample['target_seq']}'")
            print("------------------")
        return {'val_loss':metrics["BLEU"], 'log':metrics}
            
