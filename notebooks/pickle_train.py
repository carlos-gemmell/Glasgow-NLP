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


parser = argparse.ArgumentParser(description='Simply train a model using Pytorch Lightning using a pickle file.')
parser.add_argument('--model_and_dataloader_file', type=str)
parser.add_argument('--save_name', type=str)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--save_folder', type=str)
args = parser.parse_args()

load_obj = torch.load(args.model_and_dataloader_file)
model = load_obj['model']
train_dataset = load_obj['train_dataset']
train_dataloader = train_dataset.to_dataloader(args.batch_size)

trainer = Trainer(gpus=1, gradient_clip_val=0.5, amp_level='O1')
trainer.fit(model, train_dataloader)