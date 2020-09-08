from src.RawDataLoaders import *
from src.models_and_transforms.text_transforms import *
from src.models_and_transforms.BART_models import *
from src.pipe_datasets import *

from pytorch_lightning import Trainer, Callback, seed_everything
import torch

import argparse
import random
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os

'''
############ Examples ##############

Saving model and datatset too pickled file: torch.save({'model':model, 'train_dataset':train_dataset}, './BART_600_chars.pickle')

Running script using file: 
python3 pickle_train.py --model_and_dataloader_file ./BART_6k_chars.pickle --save_prefix BART_600 --batch_size 4 --save_folder ./saved_models/BART_CodeSearchNet/

###################################
'''

parser = argparse.ArgumentParser(description='Simply train a model using Pytorch Lightning using a pickle file.')
parser.add_argument('--model_and_dataloader_file', type=str)
parser.add_argument('--save_prefix', type=str)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--save_folder', type=str)
args = parser.parse_args()

load_obj = torch.load(args.model_and_dataloader_file)
model = load_obj['model']
train_dataset = load_obj['train_dataset']
train_dataloader = train_dataset.to_dataloader(args.batch_size)

class Saving_Calllback(Callback):
    def __init__(self, filepath, prefix='', epoch_save_interval=0.5):
        self.epoch_save_interval = epoch_save_interval
        self.next_save_batch_idx = 0
        self.filepath = filepath
        self.prefix = prefix
        
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        
    def on_batch_end(self, trainer, pl_module):
        if trainer.batch_idx >= self.next_save_batch_idx:
            if trainer.limit_train_batches:
                total_steps = len(trainer.train_dataloader) * trainer.limit_train_batches
            else: 
                total_steps = len(trainer.train_dataloader)
            
            filepath=os.path.join(self.filepath, f"{self.prefix}_step_{trainer.batch_idx}.ckpt")
            trainer.save_checkpoint(filepath)
            self.next_save_batch_idx += total_steps*self.epoch_save_interval
saving_cb = Saving_Calllback(filepath=args.save_folder,
                             prefix=args.save_prefix,
                             epoch_save_interval=0.05)

trainer = Trainer(gpus=[1,2], num_nodes=1, distributed_backend='ddp', gradient_clip_val=0.5, amp_level='O1', callbacks=[saving_cb])
trainer.fit(model, train_dataloader)