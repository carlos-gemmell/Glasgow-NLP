from models_and_trainers.copy_gen_transformer import CopyGeneratorModel

from torchtext.data import Field, BucketIterator
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext
import time
import math

class Model_Wrapper():
    def __init__(self, model_class, vocab, data2Dataset_fn, train_step_fn, evaluate_fn, batch_size, can_eval=True):
        self.vocab = vocab
        self.model = model_class.model
        self.data2Dataset_fn = data2Dataset_fn
        self.train_step_fn = train_step_fn
        self.evaluate_fn = evaluate_fn
        self.batch_size = batch_size
        self.can_eval = can_eval
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.optimizer = model_class.optimizer
        self.scheduler = model_class.scheduler


    def train(self, data, batch_size, steps, log_interval=200, eval_interval=1000, save_iinterval=2000):
        learning_interval=4000
        dataset, metadata = self.data2Dataset_fn(data)      

        train_iterator = BucketIterator(
            dataset,
            batch_size = batch_size,
            repeat=True,
            shuffle=True,
            sort_key = lambda x: len(x.src)+len(x.tgt),
            device = self.device)
        
        self.model.train() # Turn on the train mode
        total_loss = 0.
        start_time = time.time()
        step = 1
        for batch in train_iterator:
            loss = self.train_step_fn(batch, metadata)
            total_loss += loss.item()

            if step % log_interval == 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                print('| {:5d}/{:5d} steps | '
                      'lr {:02.4f} | ms/batch {:5.2f} | '
                      'loss {:5.2f} | ppl {:8.2f}'.format(
                        step, steps, scheduler.get_lr()[0],
                        elapsed * 1000 / log_interval,
                        cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()

            if step % eval_interval == 0 and self.can_eval:
                if self.eval__data == None:
                    print("WARNING: No evaluation data passed for in training evaluation!")
                else:
                    print("Evaluating model")
                    self.evaluate()
                    model.train()

            if step % learning_interval == 0:
                scheduler.step()

            step += 1
            if step >= steps:
                print("Finished training")
    
    def pass_eval_data(data):
        self.eval__data = data
        
    def evaluate():
        dataset, metadata = self.data2Dataset_fn(self.eval__data) 
        model.eval() # Turn on the evaluation mode
        valid_iterator = BucketIterator(dataset,
            batch_size = self.batch_size,
            sort_key = lambda x: len(x.src)+len(x.tgt),
            device = self.device)
        
        with torch.no_grad():
            score = self.evaluate_fn(valid_iterator, metadata)
            print(f"EVALUATION SCORE: {score:.2f}")
