from src.RawDataLoaders import *
from src.models_and_transforms.text_transforms import *
from src.models_and_transforms.BART_models import *
from src.pipe_datasets import *
from src.useful_utils import Validate_and_Save_Callback

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
import torch
import wandb

import argparse
import random
import itertools
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm 
import cloudpickle
import os

'''
############ Examples ##############

Saving model and datatset too pickled file: 
>>> torch.save({'model':model, 'train_dataset':train_dataset}, './BART_600_chars.pickle')

Running script using file: 
python3 pickle_train.py --pickle_file ./pickle_train_files/BART_codeBPE_CoNaLa_from_scratch.pickle --save_prefix BAT_CoNaLa_codeBPE --batch_size 32 --save_folder ./saved_models/BART_CoNaLa/ --save_interval 1.0

###################################
'''

parser = argparse.ArgumentParser(description='Simply train a model using Pytorch Lightning using a pickle file.')
parser.add_argument('--pickle_file', type=str)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--accumulate_grad', type=int, default=1)
parser.add_argument('--gpus', nargs='+', type=int, default=[0])
args = parser.parse_args()

load_obj = torch.load(args.pickle_file)
model = load_obj['model']
train_dataset = load_obj['train_dataset']
callbacks = load_obj['callbacks']
wandb_id = load_obj['wandb_id']
model_name = load_obj['model_name']
project_name = load_obj['project_name']

train_dataloader = train_dataset.to_dataloader(args.batch_size)

logger = WandbLogger(name=model_name,project=project_name, id=wandb_id)

backend = 'ddp' if len(args.gpus)>1 else None

trainer = Trainer(gpus=args.gpus, num_nodes=1, distributed_backend=backend, gradient_clip_val=0.5, amp_level='O1', callbacks=callbacks, logger=logger,
                  accumulate_grad_batches=args.accumulate_grad)
trainer.fit(model, train_dataloader)