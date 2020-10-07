from src.RawDataLoaders import *
from src.models_and_transforms.text_transforms import *
from src.models_and_transforms.BART_models import *
from src.pipe_datasets import *
from src.useful_utils import Saving_Callback

from pytorch_lightning import Trainer, seed_everything
import torch

import argparse
import random
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os

'''
############ Examples ##############

Saving model and datatset too pickled file: 
>>> torch.save({'model':model, 'train_dataset':train_dataset}, './BART_600_chars.pickle')

Running script using file: 
python3 pickle_train.py --model_and_dataloader_file ./pickle_train_files/BART_codeBPE_CoNaLa_from_scratch.pickle --save_prefix BAT_CoNaLa_codeBPE --batch_size 32 --save_folder ./saved_models/BART_CoNaLa/ --save_interval 1.0

###################################
'''

parser = argparse.ArgumentParser(description='Simply train a model using Pytorch Lightning using a pickle file.')
parser.add_argument('--model_and_dataloader_file', type=str)
parser.add_argument('--save_prefix', type=str)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--save_folder', type=str)
parser.add_argument('--save_interval', type=float, default=0.25)
parser.add_argument('--gpus', type=list, default=[1])
args = parser.parse_args()

load_obj = torch.load(args.model_and_dataloader_file)
model = load_obj['model']
train_dataset = load_obj['train_dataset']
train_dataloader = train_dataset.to_dataloader(args.batch_size)
            
saving_cb = Saving_Callback(filepath=args.save_folder,
                             prefix=args.save_prefix,
                             epoch_save_interval=args.save_interval)

trainer = Trainer(gpus=args.gpus, num_nodes=1, distributed_backend='ddp', gradient_clip_val=0.5, amp_level='O1', callbacks=[saving_cb])
trainer.fit(model, train_dataloader)