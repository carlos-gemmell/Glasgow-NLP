import argparse
from utils.file_ops import corpus_to_array, array_to_corpus
import os

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def create_folds(k_folds, src_fp, tgt_fp, out_dir, file_prefix):
    """
    This function creates k folds from a single set of input output examples. 
    each fold is sized according to the number cuts and the output consists of multiple train/test splits according to each fold.
    args:
    - k_folds: int, number of folds
    - src_fp: string, name of src input file
    - tgt_fp: string, name of tgt input file
    - out_dir: string, directory to put all the folds
    
    example:
    >>> create_folds(3, "all.desc", "all.code", "django_folds", "django")
    """
    lines = corpus_to_array(src_fp, tgt_fp)
    fold_size = int(len(lines)/k_folds) + 1
    
    folds = list(chunks(lines, fold_size))
    
    try:
        os.mkdir(args.out_dir)
    except:
        pass
    
    for i in range(k_folds):
        train_lines = []
        
        for j in range(k_folds):
            if i == j:
                test_src_name = os.path.join(args.out_dir, f'{file_prefix}.fold{i+1}-{k_folds}.test.src')
                test_tgt_name = os.path.join(args.out_dir, f'{file_prefix}.fold{i+1}-{k_folds}.test.tgt')
                array_to_corpus(folds[j], test_src_name, test_tgt_name)
            else:
                train_lines += folds[j]
        train_src_name = os.path.join(args.out_dir, f'{file_prefix}.fold{i+1}-{k_folds}.train.src')
        train_tgt_name = os.path.join(args.out_dir, f'{file_prefix}.fold{i+1}-{k_folds}.train.tgt')
        array_to_corpus(train_lines, train_src_name, train_tgt_name)
        
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='create k test train splits for a input output file pair')
    parser.add_argument('--k_folds', type=int, help='Number of folds for the dataset')
    parser.add_argument('--src_fp', help='file with source samples')
    parser.add_argument('--tgt_fp', help='file with target samples')
    parser.add_argument('--file_prefix', help='file_prefix')
    parser.add_argument('--out_dir', help='output_directory')
    args = parser.parse_args()
    
    create_folds(args.k_folds, args.src_fp, args.tgt_fp, args.out_dir, args.file_prefix)
    print(f"Finished creating {args.k_folds} folds")
    print(f"Output at: {args.out_dir}")