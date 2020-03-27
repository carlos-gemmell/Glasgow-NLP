from src.autoregressive_transformer import AutoregressiveTransformer
from src.dataset_loaders import SRC_TGT_pairs
from src.vocab_classes import Shared_Vocab, BERT_Vocab
from src.useful_utils import string_split_v3, string_split_v2, string_split_v1
from src.trainers import Model_Trainer
from src.retrieval import PyLuceneRetriever, OracleBLEURetriever
import torch
import torch.nn as nn
from torchtext.data import Field, BucketIterator
import numpy as np
import dotmap
from collections import Counter
import argparse
# from src.retrieval_decoder import retrieval_output_nudging_creator

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--dataset_config', type=str, default="django")
parser.add_argument('--retrieval_param_idx', type=int, default=0)

args = parser.parse_args()

if args.dataset_config == "django":
    src_train_fp = "datasets/django_folds/django.fold1-10.full_train.src"
    tgt_train_fp = "datasets/django_folds/django.fold1-10.full_train.tgt"
    src_valid_fp = "datasets/django_folds/django.fold1-10.valid.src"
    tgt_valid_fp = "datasets/django_folds/django.fold1-10.valid.tgt"
    src_test_fp = "datasets/django_folds/django.fold1-10.test.src"
    tgt_test_fp = "datasets/django_folds/django.fold1-10.test.tgt"

    model_save_file = "django-tiny-transformer-custom-tok-50-seq-len-850-vocab/model_file_step_495000.torch"
    output_dir = "django-retrieval-testing"
    max_seq_len = 50
    
    ret_params = [(False, 0.000285792668397205, 14, 15, 96.01759375086385, 17)]
elif args.dataset_config == "hearthstone":
    src_train_fp = "datasets/HS/train_hs.in"
    tgt_train_fp = "datasets/HS/train_hs.out"
    src_valid_fp = "datasets/HS/valid_hs.in"
    tgt_valid_fp = "datasets/HS/valid_hs.out"
    src_test_fp = "datasets/HS/test_hs.in"
    tgt_test_fp = "datasets/HS/test_hs.out"

    model_save_file = "hearthstone-tiny-transformer-valid-testing/model_file_step_195000.torch"
    output_dir = "hearthstone-retrieval-testing"
    max_seq_len = 400
    ret_params = [(False, 0.0002, 14, 15, 96.0, 17)]
elif args.dataset_config == "conala":
    src_train_fp = "datasets/CoNaLa/conala-full-train.src"
    tgt_train_fp = "datasets/CoNaLa/conala-full-train.tgt"
    src_valid_fp = "datasets/CoNaLa/conala-valid.src"
    tgt_valid_fp = "datasets/CoNaLa/conala-valid.tgt"
    src_test_fp = "datasets/CoNaLa/conala-test.src"
    tgt_test_fp = "datasets/CoNaLa/conala-test.tgt"

    model_save_file = "conala-tiny-transformer-custom-tok-75-seq-len-850-vocab/model_file_step_495000.torch"
    output_dir = "conala-retrieval-testing"
    max_seq_len = 75
    ret_params = [(False, 0.0003, 10, 10, 96.0, 17),
                  (True, 0.0002, 14, 15, 96.0, 17)]
    
vocab_size = 850
embed_dim = 512
att_heads = 4
layers = 2
batch_size = 32
dim_feedforward = 1024

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_samples = SRC_TGT_pairs(src_train_fp, tgt_train_fp, max_seq_len=max_seq_len).samples
valid_samples = SRC_TGT_pairs(src_valid_fp, tgt_valid_fp, max_seq_len=max_seq_len).samples
test_samples = SRC_TGT_pairs(src_test_fp, tgt_test_fp, max_seq_len=max_seq_len).samples

vocab = Shared_Vocab(train_samples, vocab_size, string_split_v3, use_OOVs=True)

model = AutoregressiveTransformer(vocab_size=vocab_size, embed_dim=embed_dim, att_heads=att_heads, \
                                  layers=layers, dim_feedforward=dim_feedforward, max_seq_length=max_seq_len).to(device)
test_dataset = model.data2dataset(test_samples, vocab)
test_iterator = BucketIterator(
    test_dataset,
    batch_size = batch_size,
    sort=True,
    sort_key = model.sample_order_fn,
    device = device)

valid_dataset = model.data2dataset(valid_samples, vocab)
valid_iterator = BucketIterator(
    valid_dataset,
    batch_size = batch_size,
    sort=True,
    sort_key = model.sample_order_fn,
    device = device)

model.load_model(model_save_file)
trainer = Model_Trainer(model, vocab, test_iterator=test_iterator)
outputs = trainer.evaluate(test_iterator)
avg_BLEU = np.average([out["BLEU"] for out in outputs])
print(f"Small Copy Transformer BLEU: {avg_BLEU*100:.2f}")

def retrieval_output_nudging_creator(oracle=False, relevance_interpol=0.0005, k_docs=10, k_words=10, peak_scaling_factor=40.0, \
                                     num_stop_words=10, verbose=False):
    src_train_samples = [src for src, tgt in train_samples]
    tgt_train_samples = [tgt for src, tgt in train_samples]
    if oracle:
        retriever = OracleBLEURetriever(ids_to_keep=k_docs)
        retriever.add_multiple_docs(train_samples)
    else:
        retriever = PyLuceneRetriever()
        retriever.add_multiple_docs(src_train_samples)
    
    counts = Counter(string_split_v3(" ".join(tgt_train_samples))).most_common(num_stop_words)
    stop_words = [x[0] for x in counts]
    
    def nudge_fn(last_token_log_probs, single_decoder_input, batch_encoder_ids, batch_decoder_truth_ids, OOVs):
        OOVs = OOVs.cpu().tolist()
        src_sent = vocab.decode_input(batch_encoder_ids, OOVs, copy_marker="")
        tgt_sent = vocab.decode_output(batch_decoder_truth_ids, OOVs, copy_marker="")
        current_pred = vocab.decode_output(single_decoder_input, OOVs, copy_marker="")
        top_5_ids = torch.argsort(last_token_log_probs.cpu(), descending=True)[:5]
        top_5_words = [vocab.decode_output([idx], OOVs, copy_marker="") for idx in top_5_ids]
        if verbose:
            print("## DECODE STEP ##")
            print(f"SRC input:      {src_sent}")
            print(f"TGT truth:      {tgt_sent}")
            print(f"decoded so far: {current_pred}")
            print(f"top words     : {' | '.join(top_5_words)}")
            print()
        if oracle:
            doc_ranking = retriever.search(src_sent, tgt_sent, max_retrieved_docs=k_docs)
        else:
            doc_ranking = retriever.search(src_sent, max_retrieved_docs=k_docs)
            
        retrieved_samples = [(tgt_train_samples[doc_id], score) for doc_id, score in doc_ranking]
        scoring_dict = {}
        for sample, score in retrieved_samples:
            if verbose:
                print(f"DOC: {sample}")
            sample_toks = string_split_v3(sample)
            for tok in sample_toks:
                if tok in scoring_dict:
                    scoring_dict[tok] += (peak_scaling_factor * score)/len(sample_toks)
                else:
                    scoring_dict[tok] = (peak_scaling_factor * score)/len(sample_toks)
        top_retrieved_words = [tok for tok in sorted(scoring_dict.items(), key=lambda item: -item[1]) if tok[0] not in stop_words][:k_words]
        if verbose:
            print(f"RETRIEVAL top words: {[tok for tok, score in top_retrieved_words]}")
            print()
            print()
        top_retrieved_ids = [(vocab.encode_output(tok, OOVs)[0], score) for tok, score in top_retrieved_words]
        top_retrieved_ids = [(i, s) for i, s in top_retrieved_ids if i != vocab.UNK]
        
        relevance_vector = torch.zeros_like(last_token_log_probs).fill_(-5000.0)
        for idx, score in top_retrieved_ids:
            if idx not in single_decoder_input:
                relevance_vector[idx] = score
        relevance_vector.softmax(-1)
        
        if top_5_ids[0] == vocab.EOS:
            new_probs = last_token_log_probs
        else:
            new_probs = (1-relevance_interpol) * last_token_log_probs + relevance_interpol * relevance_vector
        
        new_top_pred = torch.argmax(new_probs)
        if verbose:
            if top_5_ids[0] != new_top_pred:
                print("Relevance impact:")
                print(f"SRC input:      {src_sent}")
                print(f"TGT truth:      {tgt_sent}")
                print(f"decoded so far: {current_pred}")
                print(f"RETRIEVAL top words: {[tok for tok, score in top_retrieved_words][:]}")
                print(f"Prerdicted {vocab.decode_output([new_top_pred], OOVs)} over {top_5_words[0]}")
                print()
            
        return new_probs
    
    return nudge_fn


retrieval_decoder_fn = retrieval_output_nudging_creator(oracle=ret_params[args.retrieval_param_idx][0], 
                                                        relevance_interpol=ret_params[args.retrieval_param_idx][1], 
                                                        k_docs=ret_params[args.retrieval_param_idx][2], 
                                                        k_words=ret_params[args.retrieval_param_idx][3],  
                                                        peak_scaling_factor=ret_params[args.retrieval_param_idx][4], 
                                                        num_stop_words=ret_params[args.retrieval_param_idx][5])

model = AutoregressiveTransformer(vocab_size=vocab_size, embed_dim=embed_dim, att_heads=att_heads, \
                                  layers=layers, dim_feedforward=dim_feedforward, max_seq_length=max_seq_len, \
                                  output_nudge_fn=retrieval_decoder_fn).to(device)
model.eval()
model.load_model(model_save_file)
trainer = Model_Trainer(model, vocab, output_dir=output_dir)
outputs = trainer.evaluate(test_iterator, save_file=f"eval_test_samples_params_{args.retrieval_param_idx}.txt")
avg_BLEU = np.average([out["BLEU"] for out in outputs])
print(f"Relevance Transformer BLEU: {avg_BLEU*100:.2f}")