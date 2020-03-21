from src.autoregressive_transformer import AutoregressiveTransformer
from src.dataset_loaders import SRC_TGT_pairs
from src.vocab_classes import Shared_Vocab, BERT_Vocab
from src.useful_utils import string_split_v3, string_split_v2, string_split_v1
from src.trainers import Model_Trainer
from src.retrieval import PyLuceneRetriever
import torch
import torch.nn as nn
from torchtext.data import Field, BucketIterator
import argparse

# a default run: python3 main_autoReg.py --src_train_fp datasets/django_folds/django.fold1-10.train.src --tgt_train_fp datasets/django_folds/django.fold1-10.train.tgt --src_test_fp datasets/django_folds/django.fold1-10.test.src --tgt_test_fp datasets/django_folds/django.fold1-10.test.tgt --eval_file django_small_transformer.txt

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--max_seq_len', type=int, default=50)
parser.add_argument('--vocab_size', type=int, default=850)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--embed_dim', type=int, default=512)
parser.add_argument('--att_heads', type=int, default=8)
parser.add_argument('--layers', type=int, default=4)
parser.add_argument('--dim_feedforward', type=int, default=1024)
parser.add_argument('--steps', type=int, default=500000)
parser.add_argument('--save_interval', type=int, default=20000)
parser.add_argument('--eval_interval', type=int, default=10000)
parser.add_argument('--log_interval', type=int, default=100)
parser.add_argument('--lr', type=int, default=0.005)
# parser.add_argument('--eval_file', type=str, default="model_eval_outputs.txt")
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--src_train_fp', type=str, default="foo.txt")
parser.add_argument('--tgt_train_fp', type=str, default="foo.txt")
parser.add_argument('--src_test_fp', type=str, default="foo.txt")
parser.add_argument('--tgt_test_fp', type=str, default="foo.txt")
parser.add_argument('--concat_retrieval_to_src', type=bool, default=False)
parser.add_argument('--concat_src', type=bool, default=False)
parser.add_argument('--concat_tgt', type=bool, default=True)
parser.add_argument('--tokenization', type=str, default="custom")
parser.add_argument('--from_pretrained', type=bool, default=False)
parser.add_argument('--sort_train_dataset', type=bool, default=False)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == "cuda":
    torch.cuda.set_device(0) # choose GPU from nvidia-smi 
print("Using:", device)

train_samples = SRC_TGT_pairs(args.src_train_fp, args.tgt_train_fp, max_seq_len=args.max_seq_len).samples

test_samples = SRC_TGT_pairs(args.src_test_fp, args.tgt_test_fp, max_seq_len=args.max_seq_len).samples

if args.concat_retrieval_to_src:
    retriever = PyLuceneRetriever()
    src_train_data = [src for src,tgt in train_samples]
    retriever.add_multiple_docs(src_train_data)
    
    test_samples = retriever.augment_src_with_retrieval(test_samples, train_samples, rank_return=0, add_src=args.concat_src, add_tgt=args.concat_tgt)
    train_samples = retriever.augment_src_with_retrieval(train_samples, train_samples, rank_return=1, add_src=args.concat_src, add_tgt=args.concat_tgt)

if args.tokenization == "custom":
    vocab = Shared_Vocab(train_samples, args.vocab_size, string_split_v3, use_OOVs=True)
elif args.tokenization == "BERT":
    assert args.vocab_size == 30522, "BERT tokenization requires specific vocab size"
    vocab = BERT_Vocab()
else:
    raise NameError("invalid tokenization method specified")

model = AutoregressiveTransformer(vocab_size=args.vocab_size, embed_dim=args.embed_dim, att_heads=args.att_heads, \
                                  layers=args.layers, dim_feedforward=args.dim_feedforward, max_seq_length=args.max_seq_len, \
                                  pretrained_encoder=args.from_pretrained, pretrained_decoder=args.from_pretrained).to(device)
model.init_train_params(vocab, lr=args.lr)

train_dataset = model.data2dataset(train_samples, vocab)
test_dataset = model.data2dataset(test_samples, vocab)

train_iterator = BucketIterator(
    train_dataset,
    batch_size = args.batch_size,
    repeat=True,
    shuffle=True,
    sort=args.sort_train_dataset,
    sort_key = model.sample_order_fn if args.sort_train_dataset else None,
    device = device)

test_iterator = BucketIterator(
    test_dataset,
    batch_size = args.batch_size,
    sort=True,
    sort_key = model.sample_order_fn,
    device = device)

trainer = Model_Trainer(model, vocab, test_iterator=test_iterator, output_dir=args.output_dir)

train_logs = trainer.train(model, train_iterator, args.steps, log_interval=args.log_interval, \
                           eval_interval=args.eval_interval, save_interval=args.save_interval)

print(train_logs)