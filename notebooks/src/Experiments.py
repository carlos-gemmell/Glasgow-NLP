from abc import ABC, abstractmethod
import torch
from .metrics import nltk_bleu
import numpy as np
import os
import sys
from .useful_utils import string_split_v3, string_split_v1, chunks
import tqdm
def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')
if is_interactive():
    import tqdm.notebook as tqdm 

class Experiment(ABC):
    def __init__(self, task_data):
        """
        task_data: [(str, str)]: input/target pairs for translation evaluation.
        """
        self.task_data = task_data
    
    @abstractmethod
    def evaluate(self, prediction_fn):
        """
        This function should compute all relevant metrics to the task,
        prediction_fn: (inp) -> (pred): it's an end-to-end prediction function from any model.
        
        returns: dict: metrics
        """
        pass
        
    def save(self, path):
        """
        Saves the entire object ready to be loaded.
        """
        torch.save(self, path)
        
    def load(path):
        """
        STATIC METHOD
        accessed through class, loads a pre-existing experiment.
        """
        return torch.load(path)
    
    
class TranslationExperiment(Experiment):
    def __init__(self, task_data, src_splitter=string_split_v1, tgt_splitter=string_split_v1):
        """
        task_data: [(str, str)]: this is the expected data format.
        
        >>> from src.Experiments import TranslationExperiment
        >>> translation_experiment = TranslationExperiment(validation_pairs)
        >>> def simple_translate(src):
        >>>     return "return output"
        >>> translation_experiment.evaluate(simple_translate)
            {'BLEU': 1.4384882092392364e-09}
        """
        super().__init__(task_data)
        self.src_splitter = src_splitter
        self.tgt_splitter = tgt_splitter
        
        
    def evaluate(self, prediction_fn, save_dir=None, save_name="translation_eval.txt", batched=None):
        """
        Produces evaluation scores and saves the results to a file. The tokenisation is done through string_split_v1.
        So any non spaced text will be considered as one token.
        
        prediction_fn: (str)->(str) or [str]->[str]
        save_dir: str: folder to save the file
        save_name: str: name of file
        batched: int or None: size to use for the prediction function
        """
        if batched:
            src_sents = [src for (src, tgt) in self.task_data]
            chunked_sents = list(chunks(src_sents, batched))
            predictions = [prediction_fn(sents) for sents in tqdm.tqdm(chunked_sents, desc="predicting", total=len(chunked_sents))]
            predictions = [val for sublist in predictions for val in sublist] # flattening
        else:
            predictions = [prediction_fn(src) for (src, tgt) in tqdm.tqdm(self.task_data, desc="predicting")]
        
        # BLEU calculation
        BLEU_scores = []
        for (src, tgt), pred in tqdm.tqdm(list(zip(self.task_data, predictions)), desc="calculating bleu"):
            BLEU_score = nltk_bleu(self.tgt_splitter(tgt), self.tgt_splitter(pred))
            BLEU_scores.append(BLEU_score)
        total_BLEU = np.average(BLEU_scores)
            
        # Write to file
        if save_dir != None:
            save_path = os.path.join(save_dir, save_name)
            print(f"saving translation eval to file: {save_path}")
            with open(save_path, "w", encoding="utf-8") as out_fp:
                for (src, tgt), pred, BLEU in zip(self.task_data, predictions, BLEU_scores):

                    out_fp.write("SRC  :" + src + "\n")
                    out_fp.write("TGT  :" + tgt + "\n")
                    out_fp.write("PRED :" + pred + "\n")
                    out_fp.write("BLEU :" + str(BLEU) + "\n")
                    out_fp.write("\n")

                out_fp.write("\n\n| EVALUATION | BLEU: {:5.2f} |\n".format(total_BLEU))
                print("| EVALUATION | BLEU: {:5.3f} |".format(total_BLEU))
            
        return {"BLEU":total_BLEU}
            