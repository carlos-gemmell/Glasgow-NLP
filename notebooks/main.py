import argparse

def main(args):
    
    if args.text_tokenisation_fn = "v3": args.text_tokenisation_fn = 
    
    if args.sample_acquire_type = "src_tgt":
        if not args.no_train:
            train_samples = SRC_TGT_pairs(args.src_train_fp, args.tgt_train_fp, max_seq_len=args.max_seq_length).samples
        if not args.no_eval:
            test_samples = SRC_TGT_pairs(args.src_test_fp, args.tgt_test_fp).samples
    
    
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='create k test train splits for a input output file pair')
    parser.add_argument('--src_train_fp', help='train file with source samples')
    parser.add_argument('--tgt_train_fp', help='train file with target samples')
    parser.add_argument('--src_test_fp', help='test file with source samples')
    parser.add_argument('--tgt_test_fp', help='test file with target samples')
    
    parser.add_argument('--no_train', type=bool, default=False, help='whether the model is not trained after initialisation')
    parser.add_argument('--no_eval', type=bool, default=False, help='whether the model is not evaluated')
    
    parser.add_argument('--experiment_prefix', help='name to be appended to identify the experiment')
    parser.add_argument('--out_dir', help='output_directory')
    parser.add_argument('--restore_from', help='file containing all python opbects necessary for a full restoration of the model and vocab')
    
    parser.add_argument('--model_type', help='the kind of model that learns or infers from the data: "transformer", "copy_transformer", "BM25"')
    parser.add_argument('--sample_acquire_type', help='the kind of process that processes the data: "src_tgt"')
    parser.add_argument('--text_to_id_type', help='which way to convert text into ids: "copy", "standard"')
    
    # hyperpararms
    parser.add_argument('--lr', type=int, default=0.5, help='learning rate')
    parser.add_argument('--train_steps', type=int, default=1000000, help='Number of training steps')
    parser.add_argument('--eval_beam_size', type=int, default=1, help='eval_beam_size')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for trarining and inference')
    parser.add_argument('--log_interval', type=int, default=400, help='printing frerquency')
    parser.add_argument('--model_layers', type=int, default=4, help='model_layers')
    parser.add_argument('--model_att_heads', type=int, default=8, help='model_att_heads')
    parser.add_argument('--model_embed_dim', type=int, default=512, help='model_embed_dim')
    parser.add_argument('--model_dim_feedforward', type=int, default=1024, help='model_dim_feedforward')
    parser.add_argument('--model_att_mask_noise', type=float, default=0.0, help='model_att_mask_noise')
    
    parser.add_argument('--text_tokenisation_fn', default="v3", help='text_tokenisation_fn')
    parser.add_argument('--vocab_size', type=int, help='vocab_size')
    parser.add_argument('--max_seq_length', type=int, help='max_seq_length')
    
    
    args = parser.parse_args()
    
    main(args)