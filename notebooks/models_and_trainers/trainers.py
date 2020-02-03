from models_and_trainers.copy_gen_transformer import CopyGeneratorModel

from torchtext.data import Field, BucketIterator
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext
import time
import math
import tqdm.notebook as tqdm 

class Model_Trainer():
    def __init__(self, optimizer, scheduler):
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(self, model, iterator, train_step_fn, steps, test_iterator=None, eval_fn=None, log_interval=10, eval_interval=1000, save_interval=2000):
        learning_interval=4000
        
        model.train() # Turn on the train mode
        total_loss = 0.
        start_time = time.time()
        step = 1
        all_scores = []
        pbar = tqdm.tqdm(iterator, total=steps)
        for batch in pbar:
            loss = train_step_fn(batch)
            total_loss += loss.item()

            if step % log_interval == 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                pbar.set_description(f"Loss:{cur_loss:5.2f}, Perplx:{math.exp(cur_loss):5.2f}")
                total_loss = 0
                start_time = time.time()

            if step % eval_interval == 0 and eval_fn:
                if test_iterator:
                    print("Evaluating model")
                    scores = eval_fn(test_iterator)
                    all_scores.append(scores)
                    model.train()
                else:
                    print("WARNING: No test data passed for in training evaluation!")

            if step % learning_interval == 0:
                self.scheduler.step()

            step += 1
            if step >= steps:
                print("Finished training")
                return all_scores
    
    def pass_eval_data(self, data):
        self.eval__data = data
        
    def evaluate(self):
        dataset, metadata = self.data2Dataset_fn(self.eval__data, self.vocab) 
        self.model.eval() # Turn on the evaluation mode
        valid_iterator = BucketIterator(dataset,
            batch_size = self.batch_size,
            sort_key = lambda x: len(x.src)+len(x.tgt),
            device = self.device)
        
        with torch.no_grad():
            score = self.evaluate_fn(valid_iterator, metadata)
            print(f"EVALUATION SCORE: {score:.3f}")
