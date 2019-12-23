import argparse
from utils.vocab_classes import Shared_Vocab
from utils.useful_utils import string_split_v3, string_split_v2, string_split_v1
from utils.dataset_loaders import SRC_TGT_pairs
from models_and_trainers.trainers import Model_Wrapper
from models_and_trainers.copy_gen_transformer import CopyGeneratorModel

def main(args):
    
    if args.text_tokenisation_fn == "v3": args.text_tokenisation_fn = string_split_v3
    elif args.text_tokenisation_fn == "v2": args.text_tokenisation_fn = string_split_v2
    elif args.text_tokenisation_fn == "v1": args.text_tokenisation_fn = string_split_v1
    else: raise Exception("EXCEPTION: tokenization fn not specified correctly")
    
    if args.train:
        if args.dataset_type == "src_tgt":
            train_samples = SRC_TGT_pairs(args.src_train_fp, 
                                          args.tgt_train_fp, 
                                          max_seq_len=args.max_seq_length).samples
        else:
            raise Exception("EXCEPTION: dataset_type not set correctly: srt_tgt")
            
        if args.share_vocabulary:
            vocab = Shared_Vocab(train_samples, 
                                 args.vocab_size, 
                                 args.text_tokenisation_fn, 
                                 use_OOVs=args.use_OOVs)
        else: raise Exception("EXCEPTION: separate vocab not implemented yet.")
        
        if args.model_type == "copy_transformer":
            model = CopyGeneratorModel(vocab,
                                       args.vocab_size + args.max_seq_length, 
                                       args.model_embed_dim, 
                                       args.model_att_heads, 
                                       args.model_layers, 
                                       args.model_dim_feedforward, 
                                       args.lr,
                                       args.max_seq_length,
                                       use_copy=True)
            model_wrapper = Model_Wrapper(model_class=model, 
                                          vocab=vocab, 
                                          data2Dataset_fn = model.data2dataset_OOV, 
                                          train_step_fn=model.train_step, 
                                          evaluate_fn=model.evaluate_iterator, 
                                          batch_size=args.batch_size, 
                                          can_eval=True)
            model_wrapper.train(train_samples, 
                                args.batch_size, 
                                args.train_steps, 
                                log_interval=200, 
                                eval_interval=1000)
        
        else:
            raise Exception(f"EXCEPTION: --model_type {args.model_type} not implemented yet.")

            
        
        
        
    if args.eval:
        test_samples = SRC_TGT_pairs(args.src_test_fp, args.tgt_test_fp).samples
    
    
    
    
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='create k test train splits for a input output file pair')
    parser.add_argument('--src_train_fp', help='train file with source samples')
    parser.add_argument('--tgt_train_fp', help='train file with target samples')
    parser.add_argument('--src_test_fp', help='test file with source samples')
    parser.add_argument('--tgt_test_fp', help='test file with target samples')
    
    parser.add_argument('--train', type=bool, default=True, help='whether the model is trained after initialisation')
    parser.add_argument('--eval', type=bool, default=True, help='whether the model is evaluated')
    
    parser.add_argument('--experiment_prefix', help='name to be appended to identify the experiment')
    parser.add_argument('--out_file', help='output file for saving')
    parser.add_argument('--restore_from', help='file containing all python opbects necessary for a full restoration of the model and vocab')
    
    parser.add_argument('--model_type', help='the kind of model that learns or infers from the data: "transformer", "copy_transformer", "BM25"')
    parser.add_argument('--dataset_type', help='the kind of process that processes the data: "src_tgt"')
    parser.add_argument('--text_to_id_type', help='which way to convert text into ids: "copy", "standard"')
    
    # hyperpararms
    parser.add_argument('--lr', type=int, default=0.005, help='learning rate')
    parser.add_argument('--train_steps', type=int, default=1000000, help='Number of training steps')
    parser.add_argument('--eval_beam_size', type=int, default=1, help='eval_beam_size')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for trarining and inference')
    parser.add_argument('--log_interval', type=int, default=400, help='printing frerquency')
    parser.add_argument('--model_layers', type=int, default=4, help='model_layers')
    parser.add_argument('--model_att_heads', type=int, default=8, help='model_att_heads')
    parser.add_argument('--model_embed_dim', type=int, default=512, help='model_embed_dim')
    parser.add_argument('--model_dim_feedforward', type=int, default=1024, help='model_dim_feedforward')
    parser.add_argument('--model_att_mask_noise', type=float, default=0.0, help='model_att_mask_noise')
    parser.add_argument('--model_copy_mech', type=int, default=True, help='Allow the model to copy tokens from the input')
    
    parser.add_argument('--text_tokenisation_fn', default="v3", help='text_tokenisation_fn')
    parser.add_argument('--vocab_size', type=int, required=True, help='vocab_size')
    parser.add_argument('--max_seq_length', type=int, required=True, help='max_seq_length')
    parser.add_argument('--use_OOVs', type=bool, default=True, help='whether to add special ids for OOV tokens or just use <UNK>')
    parser.add_argument('--share_vocabulary', type=bool, default=True, help='share_vocabulary')
    parser.add_argument('--logging', type=bool, default=True, help='whether to print relevant checkpoints in the function')
    
    
    args = parser.parse_args()
    
    main(args)