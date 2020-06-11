from abc import ABC, abstractmethod
import torch.nn as nn
import torch
import math
from src.trainers import Generic_Model_Trainer 
from src.copy_gen_transformer import CopyGeneratorTransformer

import tqdm
def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')
if is_interactive():
    import tqdm.notebook as tqdm 

class Modeling_Algorithm(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def train_step(self, batch):
        pass
    
    @abstractmethod
    def train(self, steps):
        pass
    
    @abstractmethod
    def predict(self, inputs):
        pass
    
#     @abstractmethod
    def raw_predict(self, raw_inputs):
        pass
    
#     @abstractmethod
    def save(self, path):
        pass
    
#     @abstractmethod
    def load(self, path):
        pass

class Translation_Transformer(Modeling_Algorithm):
    def __init__(self, data_processor, vocab_size=30522, embed_dim=768, att_heads=12, layers=12, \
                 dim_feedforward=3072, dropout=0.1, use_copy=True, masked_look_ahead_att=True, max_seq_length=200, \
                 pretrained_encoder=False, pretrained_decoder=False):
        self.vocab_size = vocab_size
        self.data_processor = data_processor
        self.model = CopyGeneratorTransformer(vocab_size=self.vocab_size, embed_dim=embed_dim, \
                                              att_heads=att_heads, layers=layers, \
                                              dim_feedforward=dim_feedforward, use_copy=True, \
                                              pretrained_encoder=pretrained_encoder, masked_look_ahead_att=True, \
                                              pretrained_decoder=pretrained_decoder)
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=data_processor.PAD)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.05)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.99)
        
    def train_step(self, batch):
        total_loss = 0.
        encoder_input = batch["src"].to(self.model.device)
        decoder_input = batch["tgt"][:-1].to(self.model.device)
        targets = batch["tgt"][1:].to(self.model.device)
        
        src_pad_mask = encoder_input.T==self.data_processor.PAD
        tgt_pad_mask = decoder_input.T==self.data_processor.PAD
        
        self.optimizer.zero_grad()
        output = self.model(encoder_input, decoder_input, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask)

        loss = self.criterion(output.view(-1, self.vocab_size), targets.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()
        return loss
    
    def save(self, path):
        save_object = self.model.state_dict()
        torch.save(save_object, path)
        
    def load(self, path):
        save_object = torch.load(path)
        self.model.load_state_dict(save_object)
    
    def train(self, epochs, iterator, evaluation_fn=None, save_dir=None, log_interval=10, learning_interval=1000):
        try:
            pbar = tqdm.tqdm(range(epochs))
            total_loss = 0.
            for i in pbar:
                for step, batch in enumerate(iterator):
                    loss = self.train_step(batch)
                    total_loss += (loss.item() - total_loss) * 0.1

                    if step % log_interval == 0:
                        pbar.set_description(f"Loss:{total_loss:5.2f}, Perplx:{math.exp(total_loss):5.2f}")
                        total_loss = 0

                    if step % learning_interval == 0:
                        self.scheduler.step()

            print("Finished training")
        except KeyboardInterrupt:
            print("Keyboard Interrupt!")
            
    
    def predict(self, input_tensor, current_output, beam_size=1): 
        self.model.eval()
        src_pad_mask = input_tensor.T==self.data_processor.PAD
        tgt_pad_mask = current_output.T==self.data_processor.PAD
        last_predicted_ids = self.model(input_tensor, current_output, \
                                        src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask)[-1]
        self.model.train()
        return last_predicted_ids
    
    def raw_predict(self, src_str, beam_size=1, max_len=500):
        numerical_src = self.data_processor.encode_src(src_str)
        src_tensor = torch.Tensor(numerical_src).unsqueeze(1).type(torch.LongTensor).to(self.model.device)
        pred_tensor = torch.Tensor([self.data_processor.SOS]).unsqueeze(1).type(torch.LongTensor).to(self.model.device)
        
        while len(pred_tensor) < max_len:
            pred_ids = torch.argsort(self.predict(src_tensor, pred_tensor)[0], descending=True)
            for pred_id in pred_ids.tolist():
                if self.data_processor.validate_prediction(pred_tensor.view(-1).tolist()+[pred_id]):
                    pred_tensor = torch.cat((pred_tensor,torch.tensor([[pred_id]], dtype=torch.long).to(self.model.device)))
                    break
            if self.data_processor.prediction_is_complete(pred_tensor.view(-1).tolist()):
                break
        return self.data_processor.decode_tgt(pred_tensor.view(-1).tolist())
    
    def raw_batch_predict(self, batch_strings, max_len=500):
        b_numerical_src = [self.data_processor.encode_src(src_str) for src_str in batch_strings]
        b_numerical_pred = [[self.data_processor.SOS] for _ in batch_strings]
        b_completed = [False for _ in batch_strings]
        b_samples = [{'src': src_ids, 'tgt': pred_ids} for src_ids, pred_ids in zip(b_numerical_src, b_numerical_pred)]
        
        while max([len(b_samples[i]["tgt"]) for i in range(len(b_samples))]) < max_len and not all(b_completed):
            collated_samples = self.data_processor.collate(b_samples)
            last_token_predictions = self.predict(collated_samples["src"].to(self.model.device), collated_samples["tgt"].to(self.model.device))
            
            for i in range(len(b_samples)):
                pred_ids = torch.argsort(last_token_predictions[i], descending=True)
                for pred_id in pred_ids.tolist():
                    if not b_completed[i]:
                        if self.data_processor.validate_prediction(b_samples[i]["tgt"]+[pred_id]):
                            b_samples[i]["tgt"].append(pred_id)
                            if self.data_processor.prediction_is_complete(b_samples[i]["tgt"]):
                                b_completed[i] = True
                            break
        return [self.data_processor.decode_tgt(b_samples[i]["tgt"]) for i in range(len(b_samples))]