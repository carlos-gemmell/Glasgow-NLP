from torchtext.data import Field, BucketIterator
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext
import time
import math
import tqdm
from .useful_utils import super_print
import os
import datetime
    
def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')
if is_interactive():
    import tqdm.notebook as tqdm 
import numpy as np

class Model_Trainer():
    def __init__(self, model, vocab, test_iterator=None, output_dir=None):
        self.model = model
        self.vocab = vocab
        self.test_iterator = test_iterator
        self.model.stats["train_loss"] = []
        self.model.stats["eval_scores"] = []
        if output_dir:
            self.experiment_name = output_dir
            self.output_dir = os.path.join(os.getcwd(), output_dir)
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            log_dir = os.path.join(output_dir, "logs.txt")
            self.writer = super_print(log_dir)(print)
            self.writer(f"Writing logs to: {log_dir}")
        else:
            self.output_dir = None
            self.writer = print
            self.writer("'output_dir' not defined, training and model outputs won't be saved.")

    def train(self, model, iterator, steps, log_interval=100, eval_interval=1000, save_interval=2000, learning_interval=4000):
        
        self.model.train() # Turn on the train mode
        total_loss = 0.
        start_time = time.time()
        step = 1
#         try:
        pbar = tqdm.tqdm(iterator, total=steps)
        for batch in pbar:
            loss = self.model.train_step(batch)
            total_loss += loss.item()

            if step % log_interval == 0:
                cur_loss = total_loss / log_interval
                pbar.set_description(f"Loss:{cur_loss:5.2f}, Perplx:{math.exp(cur_loss):5.2f}")
                sumary = {"step":step, "loss":round(cur_loss, 2)}
                self.model.stats["train_loss"].append(sumary)
                total_loss = 0
                

            if step % eval_interval == 0 and self.test_iterator:
                self.writer("Evaluating model")
                scores = self.evaluate(self.test_iterator)
                score_keys = scores[0].keys()
                sumary = {"step":step}
                for key in score_keys:
                    if isinstance(scores[0][key], (int, float)):
                        avg_score = np.average([score[key] for score in scores])
                        sumary[key] = avg_score
                        self.writer(f"{key}:{avg_score:5.3f}")
                self.model.stats["eval_scores"].append(sumary)
                if self.output_dir:
                    self.model.plot_stats(os.path.join(self.output_dir, "training_plots.png"))
                elapsed = time.time() - start_time
                estimated_time_left = elapsed * ((steps - step)/step)
                self.writer(f"Step {step}/{steps}, estimated finish: {str(datetime.timedelta(seconds=estimated_time_left))}")
                model.train()
            
            if step % save_interval == 0:
                self.model.save_model(os.path.join(self.output_dir, f"model_file_step_{step}.torch"))

            if step % learning_interval == 0:
                self.model.scheduler.step()

            step += 1
            if step >= steps:
                self.writer("Finished training")
                return self.model.stats
#         except KeyboardInterrupt:
#             print("Keyboard Interrupt!")
#             return self.model.stats
        
    def evaluate(self, test_iterator, save_file="eval_samples.txt"):
        self.model.eval()
        with torch.no_grad():
            eval_scores = []
            pbar = tqdm.tqdm(enumerate(test_iterator), total=len(test_iterator))
            for i, batch in pbar:
                batch_eval_outputs = self.model.eval_step(batch, self.vocab)
                eval_scores += batch_eval_outputs
        if self.output_dir:
            self.model.save_eval_results(eval_scores, os.path.join(self.output_dir, save_file))
        return eval_scores
    
