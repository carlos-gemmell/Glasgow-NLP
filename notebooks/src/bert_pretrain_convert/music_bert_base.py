## Script defining the MusicBert model

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm.notebook import tqdm
from torch.nn.modules import LayerNorm
from torch.nn import TransformerEncoderLayer, TransformerEncoder

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EMPTY_TENSOR = torch.Tensor([]).to(device)


class PositionalEncodings(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEncodings, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        self.position_embeddings = nn.Embedding(max_len, d_model)
        self.token_type_embeddings = nn.Embedding(2, d_model)
        
        self.embedding_layer_norm = LayerNorm(d_model, eps=1e-12)
        
    def forward(self, x):
        
        input_shape = x.shape[:-1]
        seq_length = input_shape[0]

        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(1).expand(input_shape)

        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = x + position_embeddings + token_type_embeddings
        embeddings = self.embedding_layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    
    
class MusicBert(torch.nn.Module):

    def __init__(self, dim_sound, num_tokens, dim_model=768, nhead=12, name="music_bert",\
                 num_encoder_layers=12, d_feed=3072, dropout=0.1):
        super(MusicBert, self).__init__()

        self.d_model = dim_model
        self.num_tokens = num_tokens
        
        self.PATH = "models/" + name + ".pth"
        
        self.CLS = torch.tensor(np.random.rand(dim_model)*1e-2, \
                                requires_grad=True, \
                                dtype=torch.float32).to(device)
        self.SEP = torch.tensor(np.random.rand(dim_model)*1e-2, \
                                requires_grad=True, \
                                dtype=torch.float32).to(device)
        
        self.embedder = nn.Embedding(num_tokens, dim_model)
        self.sound_projector = nn.Linear(dim_sound, dim_model)
        self.position_embeddings = PositionalEncodings(dim_model, dropout)

        encoder_layer = TransformerEncoderLayer(dim_model, nhead, \
                                                dim_feedforward=d_feed, \
                                                dropout=dropout,
                                                activation='gelu')
        
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
        
        self.pretrain_sound = nn.Linear(dim_model, dim_sound)
        self.pretrain_tokens= nn.Linear(dim_model, num_tokens)

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        
    
    def save_model(self):
        torch.save(self.state_dict(), self.PATH)
        
    
    def load_model(self):
        self.load_state_dict(torch.load(self.PATH))
        
        
    def __concat_inputs(self, src_sound, src_tokens):
        assert len(src_sound) + len(src_tokens) != 0, \
            "Feed at least one sound or token sequence"
        
        # SHAPES: sound  N, B, F
        #         tokens N, B
        
        ## "Preprocess" the two sequences to the same Vector space
        if len(src_sound) > 0:
            src_sound = self.sound_projector(src_sound)
            src_sound = self.position_embeddings(src_sound)
            batch_size = src_sound.shape[1]
            
        if len(src_tokens) > 0:
            src_tokens = self.embedder(src_tokens.long())
            src_tokens = self.position_embeddings(src_tokens)
            batch_size = src_tokens.shape[1]
            
        x = src_sound
        # Adding the CLS & SEP Vector
#         tile_cls = self.CLS.repeat(1, batch_size, 1)
#         x = torch.cat((tile_cls, x), 0)
#         tile_sep = self.SEP.repeat(1, batch_size, 1)
#         x = torch.cat((x, tile_sep), 0)
        
        x = torch.cat((x, src_tokens), 0)
        
        return x
         

    def forward(self, src_sound=EMPTY_TENSOR, src_tokens=EMPTY_TENSOR):
        
        sequence = self.__concat_inputs(src_sound, src_tokens)
        output = self.encoder(sequence)
        
        return output
    
    
    def pretrain_predict(self, src_sound=EMPTY_TENSOR, src_tokens=EMPTY_TENSOR):
        
        sound_len = len(src_sound)
        #tokens_len = len(src_tokens)
        
        sequence = self.__concat_inputs(src_sound, src_tokens)
        entangled_sequence = self.encoder(sequence)
        
        processed_sound = entangled_sequence[1:sound_len+1]
        processed_tokens = entangled_sequence[sound_len+2:]
        
        return self.pretrain_sound(processed_sound), \
                self.pretrain_tokens(processed_tokens)
    
    
    def mask_sound_samples(self, x):
        batch_size = x.shape[0]
        sequence_length = x.shape[1]
        masking_elements = sequence_length//5
        
        if masking_elements < 1:
            masking_elements = 1
        
        # Try to sample from permutations
        mask = torch.rand([batch_size,masking_elements])*(sequence_length)
        # (batch, n_masks) indexex of masked vectors and also avoid CLS vector
        mask = mask.long()

        x = x.clone()
        for i in range(batch_size):
            x[i,mask[i]] = 0
            
        if len(x.shape)>2:
            emb_dim = x.shape[2]
            process_mask = mask.unsqueeze(-1).repeat(1,1,emb_dim)
        else:
            process_mask = mask
            

        return x, process_mask
    
    
    def mask_text_samples(self, x):
        batch_size = x.shape[0]
        sequence_length = x.shape[1]
        masking_elements = sequence_length//5
        
        if masking_elements < 1:
            masking_elements = 1
        
        mask = []
        
        for sample in x:
        
            sequence_length = np.count_nonzero(sample) # 0 is the PAD element
            # Try to sample from permutations
            sample_mask = np.random.rand(masking_elements)*(sequence_length)
            # (batch, n_masks) indexex of masked vectors and also avoid CLS vector
            mask.append(sample_mask.astype(np.int))

        mask = torch.tensor(mask)

        x = x.clone()
        for i in range(batch_size):
            x[i,mask[i]] = 1 # MASK index
            
        if len(x.shape)>2:
            emb_dim = x.shape[2]
            process_mask = mask.unsqueeze(-1).repeat(1,1,emb_dim)
        else:
            process_mask = mask
            

        return x, process_mask
    
    
    def pretrain_model(self, pretrain_dataloader, val_dataloader, epochs):

        self.train()
        sound_criterion = nn.MSELoss()
        tokens_criterion = nn.CrossEntropyLoss()
        lr = 1e-2 # learning rate
        pretrain_optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        pretrain_scheduler = torch.optim.lr_scheduler.StepLR(pretrain_optimizer,
                                                             1.0, gamma=0.95)
        total_loss = [0., 0., 0.]
        save_interval = int(len(pretrain_dataloader)*0.1)
        eval_interval = int(len(pretrain_dataloader)*0.1)
        
        learning_curve = []

        try:
            for n_epoch in range(epochs):
                pbar = tqdm(pretrain_dataloader,
                            desc="Train ({:d}/{:d} Epoch) - Tot_Loss XXXXXX"\
                            .format(n_epoch+1, epochs))
                for i, batch in enumerate(pbar):
                    pretrain_optimizer.zero_grad()
                    sample_sound = batch['song_features']
                    masked_sound, sound_mask = self.mask_sound_samples(sample_sound)
                    sample_tokens = batch['full_text']
                    masked_tokens, tokens_mask = self.mask_text_samples(sample_tokens)

                    output_sound, output_tokens = self.pretrain_predict(\
                                                        sample_sound.permute(1,0,2).to(device),
                                                        masked_tokens.permute(1,0).to(device))

                    masked_sound_gt = torch.gather(sample_sound.to(device), 1,
                                                   sound_mask.to(device))
                    masked_sound_out = torch.gather(output_sound.permute(1,0,2).to(device), 1,
                                                   sound_mask.to(device))

                    masked_tokens_gt = torch.gather(sample_tokens.to(device), 1,
                                                   tokens_mask.to(device))

                    masked_tokens_out = torch.gather(output_tokens.permute(1,0,2).to(device), 1,
                                                     tokens_mask.unsqueeze(-1).\
                                                     repeat(1,1,self.num_tokens).to(device))

                    sound_loss  = sound_criterion(masked_sound_gt, masked_sound_out)
                    tokens_loss = tokens_criterion(masked_tokens_out.permute(0,2,1),
                                                   masked_tokens_gt)

                    loss = tokens_loss + sound_loss/100 # Normalize for the length disparity
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                    pretrain_optimizer.step()

                    if total_loss[0] == 0:
                        total_loss = np.array([loss.item(), tokens_loss.item(), sound_loss.item()])
                    else:
                        tmp = np.array([loss.item(), tokens_loss.item(), sound_loss.item()])
                        total_loss = 0.99*total_loss + 0.01*tmp

                    if i%10 == 0:
                        pbar.set_description('Pre-train ({:d}/{:d} Epoch) - Tot_Loss {:.6f}'\
                                             .format(n_epoch+1, epochs, total_loss[0]))

                    if (i+1)%eval_interval == 0:
                        eval_loss = self.evaluate(val_dataloader)
                        learning_curve.append(eval_loss)

                    if (i+1)%save_interval == 0:
                        self.save_model()
                        print("Model saved after {:d} batches".format(i))
        except KeyboardInterrupt:
            print("KeyboardException! Exiting...")
        return learning_curve
        
        
    def evaluate(self, val_dataloader):
        self.eval()
        with torch.no_grad():
            sound_criterion = nn.MSELoss()
            tokens_criterion = nn.CrossEntropyLoss()
            
            losses = []
            pbar = tqdm(val_dataloader, desc="Evaluation: ")
            for i, batch in enumerate(pbar):
                
                sample_sound = batch['song_features']
                masked_sound, sound_mask = self.mask_sound_samples(sample_sound)
                sample_tokens = batch['full_text']
                masked_tokens, tokens_mask = self.mask_text_samples(sample_tokens)
                
                output_sound, output_tokens = self.pretrain_predict(\
                                                    sample_sound.permute(1,0,2).to(device),
                                                    masked_tokens.permute(1,0).to(device))
                
                masked_sound_gt = torch.gather(sample_sound.to(device), 1,
                                               sound_mask.to(device))
                masked_sound_out = torch.gather(output_sound.permute(1,0,2).to(device), 1,
                                               sound_mask.to(device))
                
                masked_tokens_gt = torch.gather(sample_tokens.to(device), 1,
                                               tokens_mask.to(device))

                masked_tokens_out = torch.gather(output_tokens.permute(1,0,2).to(device), 1,
                                                 tokens_mask.unsqueeze(-1).\
                                                 repeat(1,1,self.num_tokens).to(device))
                
                sound_loss  = sound_criterion(masked_sound_gt, masked_sound_out)
                tokens_loss = tokens_criterion(masked_tokens_out.permute(0,2,1),
                                               masked_tokens_gt)
                
                loss = tokens_loss + sound_loss/100 # Normalize for the length disparity
                
                losses += [loss]
                
        return float(torch.mean(torch.tensor(losses)))