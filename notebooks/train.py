import argparse

from src.dataset_loaders import CoNaLa_RawDataLoader, Django_RawDataLoader, Parseable_Django_RawDataLoader
from src.DataProcessors import Parse_Tree_Translation_DataProcessor, CodeTrainedBPE_Translation_DataProcessor
from src.translation_transformer import Translation_Transformer
from src.Experiments import TranslationExperiment
from src.tree_sitter_AST_utils import Node_Processor
import torch

parser = argparse.ArgumentParser(description='Only train a model and save to a file, nothing else.')
parser.add_argument('--dataset', type=str, default="django")
parser.add_argument('--processor', type=str, default="code_BPE")
parser.add_argument('--save_file', type=str, default="trained_model.pytorch")
parser.add_argument('--batch_sz', type=int, default=32)
parser.add_argument('--epochs', type=int, default=400)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# get the data

if args.dataset == "filtered_django":
    raw_data_loader = Parseable_Django_RawDataLoader()
    
train_pairs = raw_data_loader.english_to_code_for_translation("train")

    
# process the data
if args.processor == "parse_tree":
    processor = Parse_Tree_Translation_DataProcessor(train_pairs)
elif args.processor == "code_BPE":
    processor = CodeTrainedBPE_Translation_DataProcessor(train_pairs)

    
train_dataloader = processor.to_dataloader(args.batch_sz, num_workers=0)


# train on the processed data
translator = Translation_Transformer(processor,
                                     processor.vocab_size, 
                                     embed_dim=256, 
                                     att_heads=4, 
                                     layers=4, 
                                     dim_feedforward=1024, 
                                     max_seq_length=550,
                                     use_copy=False)
translator.model.to(device)
translator.train(args.epochs, train_dataloader)


# save the model
translator.save(args.save_file)