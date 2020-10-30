from src.RawDataLoaders import *
from src.models_and_transforms.text_transforms import *
from src.models_and_transforms.BART_models import *
from src.pipe_datasets import *
from src.useful_utils import Validate_and_Save_Callback

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
import torch

import argparse
import random
import itertools
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm 
import cloudpickle
import os
import wandb


parser = argparse.ArgumentParser(description='Simply train a model using Pytorch Lightning using a pickle file.')
parser.add_argument('--pickle_file', type=str)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--gpus', nargs='+', type=int, default=[0])
args = parser.parse_args()

load_obj = torch.load(args.pickle_file)
model = load_obj['model']
train_dataset = load_obj['train_dataset']
callbacks = load_obj['callbacks']
save_dir = load_obj['save_dir']
