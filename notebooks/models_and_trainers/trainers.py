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


    def train(self, data, batch_size, steps, log_interval=200, eval_interval=1000, save_interval=2000):
        learning_interval=4000
        dataset, metadata = self.data2Dataset_fn(data, self.vocab)      

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
                        step, steps, self.scheduler.get_lr()[0],
                        elapsed * 1000 / log_interval,
                        cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()

            if step % eval_interval == 0 and self.can_eval:
                if hasattr(self, 'eval__data'):
                    print("Evaluating model")
                    self.evaluate()
                    self.model.train()
                else:
                    print("WARNING: No evaluation data passed for in training evaluation!")

            if step % learning_interval == 0:
                self.scheduler.step()

            step += 1
            if step >= steps:
                print("Finished training")
    
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
