import torch
import torch.nn as nn
import time
import math
from tqdm.auto import tqdm 
import os
from pytorch_lightning.utilities import move_data_to_device
from pytorch_lightning.overrides.data_parallel import LightningDataParallel
    
def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')
if is_interactive():
    import tqdm.notebook as tqdm 
import numpy as np

class Model_Trainer():
    def __init__(self, gpus=[]):
        self.stats = {}
        self.stats["train_loss"] = []
        self.stats["eval_scores"] = []
        self.gpus = gpus
        
        if len(gpus)==0:
            self.device = 'cpu'
        elif len(gpus)==1:
            self.device = f"cuda:{gpus[0]}"
            print(f"Detected {torch.cuda.device_count()} GPUS available, using {gpus}.")
        elif len(gpus)>1:
            self.device = f"cuda:{gpus[0]}"
            print(f"Main device is: {self.device}")

    def train(self, model, dataloader, epochs=float("inf"), valid_interval=2, save_interval=1, dp=False):
        
        optimizer = model.configure_optimizers()
        
        model.to(self.device)
        if dp:
            model = LightningDataParallel(model, device_ids=self.gpus)
            
        
        model.eval() # Turn on the train mode
        current_epoch = 1
        try:
            while current_epoch < epochs:
                pbar = tqdm.tqdm(dataloader, total=len(dataloader))
                for idx, batch in enumerate(pbar):
                    optimizer.zero_grad()
                    batch = move_data_to_device(batch, self.device)
                    if dp:
                        loss_obj = model(batch, idx)
                        loss = loss_obj["loss"].mean()
                    else:
                        loss_obj = model.training_step(batch, idx)
                        loss = loss_obj["loss"]
                    loss.backward()
                    if torch.isnan(loss):
                        return batch
                    
                    optimizer.step()

                    pbar.set_description(f"Epoch: {current_epoch}, Loss:{loss:5.2f}")
                    self.stats["train_loss"].append(loss)

        except KeyboardInterrupt:
            print("Keyboard Interrupt!")
            return self.stats
        print("finished training")
        return self.stats
        
    def _batch_to_device(self, batch):
        if isinstance(batch, dict):
            for key in list(batch.keys()):
                batch[key].to(self.device)
            return batch
        else:
            print("not supperted yet")