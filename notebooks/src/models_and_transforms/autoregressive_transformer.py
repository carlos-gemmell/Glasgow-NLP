from .copy_gen_transformer import CopyGeneratorTransformer
from .beam_search import beam_search_decode
from .useful_utils import batch_filter_ids
from .metrics import nltk_bleu
import torch
import copy
import time
import torch.nn as nn
import torchtext
from torchtext.data import Field
import os
import matplotlib.pyplot as plt


class AutoregressiveTransformer(nn.Module):
    def __init__(self, vocab_size=30522, embed_dim=768, att_heads=12, layers=12, dim_feedforward=3072, dropout=0.1, use_copy=True, \
                 masked_look_ahead_att=True, max_seq_length=200, pretrained_encoder=False, pretrained_decoder=False, output_nudge_fn=None):
        super(AutoregressiveTransformer, self).__init__()
        self.max_seq_length = max_seq_length
        self.output_nudge_fn = output_nudge_fn
        if not use_copy or pretrained_encoder or pretrained_decoder:
            self.output_vocab_size = vocab_size
        else:
            self.output_vocab_size = vocab_size+max_seq_length
        self.model = CopyGeneratorTransformer(vocab_size=self.output_vocab_size, embed_dim=embed_dim, \
                                              att_heads=att_heads, layers=layers, \
                                              dim_feedforward=dim_feedforward, use_copy=True, \
                                              pretrained_encoder=pretrained_encoder, masked_look_ahead_att=True, \
                                              pretrained_decoder=pretrained_decoder)
        
        self.params = self.model.parameters()
        self.stats = {}
        
    def init_train_params(self, vocab, lr=0.005, gamma=0.99):
        self.criterion = nn.CrossEntropyLoss(ignore_index=vocab.PAD)
        self.optimizer = torch.optim.SGD(self.params, lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=gamma)
        
    def train_step(self, batch):
        total_loss = 0.
        start_time = time.time()
        encoder_input = batch.src
        decoder_input = batch.tgt[:-1]
        targets = batch.tgt[1:]

        self.optimizer.zero_grad()
        output = self.model(encoder_input, decoder_input)

        loss = self.criterion(output.view(-1, self.output_vocab_size), targets.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, 0.5)
        self.optimizer.step()
        elapsed = time.time() - start_time
        return loss
    
    def eval_step(self, batch, vocab):
        batch_size = batch.src.shape[1]
        eval_outputs = []
        predictions = batch_to_output_autorregressive(self.model, batch, vocab, max_seq_len=self.max_seq_length, \
                                                      output_nudge_fn=self.output_nudge_fn)

        unwanted_ids = [vocab.SOS, vocab.EOS, vocab.PAD]
        sources = batch.src.transpose(0,1).cpu().tolist()
        sources = batch_filter_ids(sources,unwanted_ids)

        predictions = [t[0].view(-1).cpu().tolist() for t in predictions]
        predictions = batch_filter_ids(predictions,unwanted_ids)

        targets = batch.tgt.transpose(0,1).cpu().tolist()
        targets = batch_filter_ids(targets,unwanted_ids)

        for j in range(batch_size):
            sample_eval_out = {}
            BLEU = nltk_bleu(targets[j], predictions[j])
            sample_eval_out["BLEU"] = BLEU
            OOV_ids = batch.OOVs.cpu()[:,j].tolist()

            sample_eval_out["SRC"] = vocab.decode_input(sources[j],OOV_ids)
            sample_eval_out["TGT"] = vocab.decode_output(targets[j],OOV_ids)
            sample_eval_out["PRED"] = vocab.decode_output(predictions[j],OOV_ids)
            eval_outputs.append(sample_eval_out)
        return eval_outputs
    
    def data2dataset(self, data, vocab):
        TEXT_FIELD = Field(sequential=True, use_vocab=False, unk_token=vocab.UNK, init_token=vocab.SOS, \
                           eos_token=vocab.EOS, pad_token=vocab.PAD)
        OOV_TEXT_FIELD = Field(sequential=True, use_vocab=False, pad_token=vocab.PAD)

        examples = []

        for (src, tgt) in data:
            src_ids, OOVs = vocab.encode_input(src)
            tgt_ids = vocab.encode_output(tgt, OOVs)
            example = torchtext.data.Example.fromdict({"src":src_ids, 
                                                   "tgt":tgt_ids, 
                                                   "OOVs":OOVs}, 
                                                        fields={"src":("src",TEXT_FIELD), 
                                                                "tgt":("tgt",TEXT_FIELD), 
                                                                "OOVs":("OOVs", OOV_TEXT_FIELD)})
            examples.append(example)
        fields = {"src":TEXT_FIELD, "tgt":TEXT_FIELD, "OOVs":OOV_TEXT_FIELD}
        dataset = torchtext.data.Dataset(examples,fields=fields)
        return dataset
    
    def sample_order_fn(self, batch):
        return len(batch.src)+len(batch.tgt)

    def plot_stats(self, output_file):
        if self.stats  == {}:
            print("empty model stats, not plotting.")
            return
        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('train steps')
        train_loss = [e['loss'] for e in self.stats["train_loss"]]
        train_loss_steps = [e['step'] for e in self.stats["train_loss"]]
        ax1.set_ylabel('train loss', color=color)
        ax1.plot(train_loss_steps,train_loss, color=color)
        
        ax2 = ax1.twinx()
        score_keys = ["BLEU"]
        for key in score_keys:
            if isinstance(self.stats["eval_scores"][0][key], (int, float)):
                x = [0]+[e['step'] for e in self.stats["eval_scores"]]
                y = [0]+[e[key] for e in self.stats["eval_scores"]]
                ax2.plot(x, y, label=key)
                ax2.legend()

        plt.savefig(output_file)
    
    def save_eval_results(self, eval_scores, output_file):
        with open(output_file, "w") as eval_f:
            for output in eval_scores:
                for key, value in output.items():
                    eval_f.write(f"{key:>12} : {value} \n")
                eval_f.write(f"\n")
    
    def save_model(self, save_file):
        torch.save((self.model.state_dict(), self.optimizer, self.scheduler, \
                    self.max_seq_length, self.output_vocab_size, self.stats), save_file)
    
    def load_model(self, save_file):
        (model_state_dict, optimizer, scheduler, \
                    max_seq_length, output_vocab_size, stats) = torch.load(save_file)
        assert max_seq_length == self.max_seq_length, "wrong max seq length, important sice copy mechanism depends on it"
        assert output_vocab_size == self.output_vocab_size, "vocab size missmatch"
        self.max_seq_length = max_seq_length
        self.output_vocab_size = output_vocab_size
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model.load_state_dict(model_state_dict)

    
def batch_to_output_autorregressive(model, batch, vocab, max_seq_len=30, output_nudge_fn=None):
    encoder_inputs = batch.src
    decooder_inputs = batch.tgt
    OOVss = batch.OOVs
    predictions = beam_search_decode(model,
                      batch_encoder_ids=encoder_inputs,
                      batch_decoder_truth_ids=decooder_inputs,
                      OOVss=OOVss,
                      output_nudge_fn=output_nudge_fn,
                      SOS_token=vocab.SOS,
                      EOS_token=vocab.EOS,
                      PAD_token=vocab.PAD,
                      beam_size=1,
                      max_length=max_seq_len,
                      num_out=1)
    return predictions