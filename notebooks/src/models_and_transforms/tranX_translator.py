from __future__ import print_function
import sys
sys.path.insert(0, "src/external_repos/tranX")

import time

import astor
import six.moves.cPickle as pickle
from six.moves import input
from six.moves import xrange as range
from torch.autograd import Variable

import evaluation
from asdl.asdl import ASDLGrammar
from asdl.transition_system import TransitionSystem
from common.utils import update_args, init_arg_parser
from components.dataset import Dataset
from components.reranker import *
from components.standalone_parser import StandaloneParser
from model import nn_utils
from model.paraphrase import ParaphraseIdentificationModel
from model.parser import Parser
from model.reconstruction_model import Reconstructor
from model.utils import GloveHelper
from exp import *

class TranX_Prertrained_Translator():
    def __init__(self, dataset="django"):
        arg_parser = init_arg_parser()
        if dataset=="django":
            args = init_arg_parser().parse_args("--mode test \
                                         --load_model src/external_repos/tranX/data/pretrained_models/django.bin \
                                         --beam_size 15 \
                                         --test_file src/external_repos/tranX/data/django/test.bin \
                                         --save_decode_to 0.test.decode \
                                         --decode_max_time_step 100 \
                                         --example_preprocessor django_example_processor".split())
        elif dataset=="conala":
            args = init_arg_parser().parse_args("--mode test \
                                         --load_model src/external_repos/tranX/data/pretrained_models/conala.bin \
                                         --beam_size 15 \
                                         --test_file src/external_repos/tranX/data/conala/test.bin \
                                         --save_decode_to 0.test.decode \
                                         --decode_max_time_step 100 \
                                         --example_preprocessor conala_example_processor".split())
            
        self.parser = StandaloneParser(args.parser,
                              args.load_model,
                              args.example_preprocessor,
                              beam_size=args.beam_size,
                              cuda=args.cuda)
    
    def raw_predict(self, src_str):
#         try:
            utterance = src_str.strip()
            hypotheses = self.parser.parse(utterance, debug=False)

            pred_code_list = [hyp.code for hyp in hypotheses]
            return pred_code_list[0]
#         except:
#             return ""
    
    def raw_batch_predict(self, batch_src_strs):
        return [self.raw_predict(src_str) for src_str in batch_src_strs]