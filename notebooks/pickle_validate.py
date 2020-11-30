from src.RawDataLoaders import *
from src.models_and_transforms.text_transforms import *
from src.models_and_transforms.BART_models import *
from src.pipe_datasets import *
from src.useful_utils import sort_nicely

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
import torch

import argparse
import random
import itertools
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm 
import cloudpickle
import os
import wandb
import time


parser = argparse.ArgumentParser(description='Simply train a model using Pytorch Lightning using a pickle file.')
parser.add_argument('--pickle_file', type=str)
parser.add_argument('--chunk_size', type=int, default=16)
parser.add_argument('--gpus', nargs='+', type=int, default=[0])
args = parser.parse_args()

load_obj = torch.load(args.pickle_file)
valid_fn = load_obj['valid_fn']
save_dir = load_obj['save_dir']
valid_metric = load_obj['valid_metric']
valid_fn_kwargs = load_obj['valid_fn_kwargs'] if 'valid_fn_kwargs' in load_obj else {}
wandb_id = load_obj['wandb_id']
model_name = load_obj['model_name']
project_name = load_obj['project_name']

valid_fn_kwargs.update({'chunk_size':args.chunk_size})

logger = WandbLogger(name=model_name,project=project_name, id=wandb_id)

while True:
    all_files = os.listdir(save_dir)
    sort_nicely(all_files)
    print(all_files)
    ckpt_files = [file_name for file_name in all_files if '.ckpt' in file_name]
    fresh_ckpt_files = [file_name for file_name in ckpt_files if 'SAVED' in file_name]
    
    if fresh_ckpt_files == []:
        time.sleep(0.5)
        continue
    
    oldest_ckpt_file = fresh_ckpt_files[0]
    print(f'Evaluating checkpoint: {oldest_ckpt_file}')
    
    processing_ckpt_file_name = oldest_ckpt_file.replace('SAVED', 'PROCESSING')
    os.rename(os.path.join(save_dir, oldest_ckpt_file), os.path.join(save_dir, processing_ckpt_file_name))
    valid_dict = valid_fn(os.path.join(save_dir, processing_ckpt_file_name), device=f'cuda:{args.gpus[0]}', **valid_fn_kwargs)
    valid_score = valid_dict[valid_metric]
    
    ckpt_dict = torch.load(os.path.join(save_dir, processing_ckpt_file_name), map_location='cpu')
    valid_dict['global_step'] = ckpt_dict['global_step']
    logger.log_metrics(valid_dict)
    
    validated_ckpt_file_name = processing_ckpt_file_name.replace('PROCESSING', f'VALID_{float(valid_score):.4f}')
    
    os.rename(os.path.join(save_dir, processing_ckpt_file_name), os.path.join(save_dir, validated_ckpt_file_name))
    print(f'Evaluation complete, renaming checkpoint to: {validated_ckpt_file_name}')
    print()