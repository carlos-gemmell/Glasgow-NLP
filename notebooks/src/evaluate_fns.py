from .useful_utils import batch_filter_ids
from .metrics import nltk_bleu
import .beam_search as beam_search
import tqdm.notebook as tqdm 
import numpy as np
import torch
import torch.nn as nn
import torchtext
from transformers import BertTokenizer

def evaluate(model, iterator, batch_to_output_fn, vocab, out_file=None, max_seq_len=30):
    model.eval() # Turn on the evaluation mode
    with torch.no_grad():
        BLEU_scores = []
        pbar = tqdm.tqdm(enumerate(iterator), total=len(iterator))
        if out_file:
            with open(out_file, "a", encoding="utf-8") as out_fp:
                pass
        for i, batch in pbar:
            batch_size = batch.src.shape[1]
            
            predictions = batch_to_output_fn(model, batch, vocab, max_seq_len=max_seq_len)
#             encoder_inputs = batch.src
#             predictions = beam_search.beam_search_decode(model,
#                               batch_encoder_ids=encoder_inputs,
#                               SOS_token=SOS_token,
#                               EOS_token=EOS_token,
#                               PAD_token=PAD_token,
#                               beam_size=1,
#                               max_length=max_seq_len,
#                               num_out=1)
            
            unwanted_ids = [vocab.SOS, vocab.EOS, vocab.PAD]
            sources = batch.src.transpose(0,1).cpu().tolist()
            sources = batch_filter_ids(sources,unwanted_ids)
            
            predictions = [t[0].view(-1).cpu().tolist() for t in predictions]
            predictions = batch_filter_ids(predictions,unwanted_ids)
            
            targets = batch.tgt.transpose(0,1).cpu().tolist()
            targets = batch_filter_ids(targets,unwanted_ids)
            
            for j in range(batch_size):
                BLEU = nltk_bleu(targets[j], predictions[j])
                BLEU_scores.append(BLEU)
                if out_file:
                    with open(out_file, "a", encoding="utf-8") as out_fp:
                        OOV_ids = batch.OOVs.cpu()[:,j].tolist()

                        out_fp.write("SRC  :" + vocab.decode_input(sources[j],OOV_ids) + "\n")
                        out_fp.write("TGT  :" + vocab.decode_output(targets[j],OOV_ids) + "\n")
                        out_fp.write("PRED :" + vocab.decode_output(predictions[j],OOV_ids) + "\n")
                        out_fp.write("BLEU :" + str(BLEU) + "\n")
                        out_fp.write("\n")
            pbar.set_description(f"BLEU:{np.average(BLEU_scores):5.2f}")
        final_BLEU = np.average(BLEU_scores)
        if out_file:
            with open(out_file, "a", encoding="utf-8") as out_fp:
                out_fp.write("\n\n| EVALUATION | BLEU: {:5.2f} |\n".format(final_BLEU))
        print("| EVALUATION | BLEU: {:5.3f} |".format(final_BLEU))
    return (final_BLEU)

def batch_to_output_autorregressive(model, batch, vocab, max_seq_len=30):
    encoder_inputs = batch.src
    predictions = beam_search.beam_search_decode(model,
                      batch_encoder_ids=encoder_inputs,
                      SOS_token=vocab.SOS,
                      EOS_token=vocab.EOS,
                      PAD_token=vocab.PAD,
                      beam_size=1,
                      max_length=max_seq_len,
                      num_out=1)
    return predictions


def create_eval_fn(model, iterator, batch_to_output_fn, vocab, out_file=None, max_seq_len=30):
    def eval_fn():
        return evaluate(model, iterator, batch_to_output_fn, vocab, out_file=out_file, max_seq_len=max_seq_len)
    return eval_fn

# def evaluate(model, test_data, output_fn=None):
#     if output_fn is not None:
#         output = output_fn(model, test_data)
#     else:
#         output = model(test_data)