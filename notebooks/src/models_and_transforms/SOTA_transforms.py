import sys
sys.path.insert(0, "src/external_repos/tranX")
from components.standalone_parser import StandaloneParser
from tqdm.auto import tqdm 


class CoNaLa_SOTA_Transform():
    def __init__(self, cuda=True, fields={'input_field':'input_text', 'output_field':'pred_text'}):
        '''
        This uses the model from Frank Xu preseented in: 
        Incorporating External Knowledge through Pre-training for Natural Language to Code Generation
        
        It translates English to Python code
        '''        
        self.fields = fields
        parser = 'default_parser'
        model_file = 'src/external_repos/external-knowledge-codegen/best_pretrained_models/finetune.mined.retapi.distsmpl.dr0.3.lr0.001.lr_de0.5.lr_da15.beam15.seed0.mined_100000.intent_count100k_topk1_temp5.bin'
        processor = 'conala_example_processor'
        beam_size = 15
        reranker_file = 'src/external_repos/external-knowledge-codegen/best_pretrained_models/reranker.conala.vocab.src_freq3.code_freq3.mined_100000.intent_count100k_topk1_temp5.bin'
        self.parser = StandaloneParser(parser,
                              model_file,
                              processor,
                              beam_size=beam_size,
                              cuda=cuda,
                              reranker_path=reranker_file)
        
    def __call__(self, samples):
        '''
        samples: [dict]: [{'input_text': "`foo` is an empty list"},...]
        returns: [dict]: [{'input_text': "`foo` is an empty list", 'pred_text': "foo = []"}]
        '''
        for sample_obj in tqdm(samples, desc='Tranx:'):
            input_text = sample_obj[self.fields['input_field']]
            input_text = input_text.strip()
            hypotheses = self.parser.parse(input_text, debug=False)
            
            sample_obj[self.fields['output_field']] = hypotheses[0].code
        return samples
    
class Django_SOTA_Transform(CoNaLa_SOTA_Transform):
    def __init__(self, cuda=True, fields={'input_field':'input_text', 'output_field':'pred_text'}):
        '''
        This uses the model from Pengchen Yin preseented in: 
        Incorporating External Knowledge through Pre-training for Natural Language to Code Generation
        
        It translates English to Python code
        ''' 
        self.fields = fields
        parser = 'default_parser'
        model_file = 'src/external_repos/external-knowledge-codegen/best_pretrained_models/finetune.mined.retapi.distsmpl.dr0.3.lr0.001.lr_de0.5.lr_da15.beam15.seed0.mined_100000.intent_count100k_topk1_temp5.bin'
        processor = 'django_example_processor'
        beam_size = 15
        self.parser = StandaloneParser(parser,
                              model_file,
                              processor,
                              beam_size=beam_size,
                              cuda=cuda)