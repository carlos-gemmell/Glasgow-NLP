from datetime import datetime
from torchtext.data import Field, BucketIterator
import re
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

def super_print(filename):
    '''
    filename is the file where output will be written
    
    Example:
    >>> print = super_print("out_logs.txt")(print)
    >>> print("See me in the logs!")
    '''
    def wrap(func):
        '''func is the function you are "overriding", i.e. wrapping'''
        def wrapped_func(*args,**kwargs):
            '''*args and **kwargs are the arguments supplied 
            to the overridden function'''
            #use with statement to open, write to, and close the file safely
            with open(filename,'a', encoding="utf-8") as outputfile:
                now = datetime.now()
                dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                outputfile.write("[{}] ".format(dt_string))
                outputfile.write(" ".join(str(x) for x in args))
                outputfile.write("\n")
            #now original function executed with its arguments as normal
            return func(*args,**kwargs)
        return wrapped_func
    return wrap


def filter_corpus(data, max_seq_length=50, tokenizer=str.split):
    '''
    This function removes all pairs of samples where either src or tgt excede the number of max tokens.
    Args:
        data: [(str,str)]: This is a list of tuples containing src and tgt pairs
    Returns:
        data: [(str,str)]: the filtered list
    '''
    return [(src, tgt) for src, tgt in data if len(tokenizer(src)) <= max_seq_length and len(tokenizer(tgt)) <= max_seq_length]

def clean_samples(data):
    return [(src.strip(), tgt.strip()) for src, tgt in data]


def string_split_v1(s):
    '''
    This will chunk all code properly, splits strings with quotes and splits variables with underscores
    Args:
        s: string, string to be split
    
    Example:
    >>> text = "create variable student_names with string 'foo bar baz'"
    >>> print(string_split_v1(text))
    ['create', 'variable', 'student', '_', 'names', 'with', 'string', "'", 'foo', 'bar', 'baz', "'"]
    '''
    return list(filter(lambda x: x != '' and x != "\n" and not x.isspace(), re.split('(_|\W)', s)))

def string_split_v2(s):
    '''
    This will chunk all code properly and splits variables with underscores, no inner quote split
    Args:
        s: string, string to be split
    
    Example:
    >>> text = "create variable student_names with string 'foo bar baz'"
    >>> print(string_split_v2(text))
    ['create', 'variable', 'student', '_', 'names', 'with', 'string', "'foo bar baz'"]
    '''
    return list(filter(lambda x: x != '' and x != "\n" and not x.isspace(), re.split('(\\\'.*?\\\'|\\\".*?\\\"|_|\W)', s)))

def string_split_v3(s):
    '''
    This will chunk all code properly. No variable splitting or quoted text splitting
    Args:
        s: string, string to be split
    
    Example:
    >>> text = "create variable student_names with string 'foo bar baz'"
    >>> print(string_split_v3(text))
    ['create', 'variable', 'student_names', 'with', 'string', "'foo bar baz'"]
    '''
    return list(filter(lambda x: x != '' and x != "\n" and not x.isspace(), re.split('(\\\'.*?\\\'|\\\".*?\\\"|\W)', s)))

def samples_to_dataset(samples):
    """
    Args:
        samples: [(src_string),(tgt_string)]
        src/tgt_tokenizer: a func that takes a string and returns an array of strings
    """
    examples = []
    TEXT_FIELD = Field(sequential=True, use_vocab=False, init_token='<sos>',eos_token='<eos>')
    
    for sample in samples:
        src_string, tgt_string = sample
        examples.append(torchtext.data.Example.fromdict({"src":src_string, "tgt":tgt_string}, 
                                        fields={"src":("src",TEXT_FIELD), "tgt":("tgt",TEXT_FIELD)}))
        
    dataset = torchtext.data.Dataset(examples,fields={"src":src_field, "tgt":tgt_field})
    return dataset

def nltk_bleu(refrence, prediction):
    """
    Implementation from ReCode
    and moses multi belu script sets BLEU to 0.0 if len(toks) < 4
    """
    ngram_weights = [0.25] * min(4, len(refrence))
    return sentence_bleu([refrence], prediction, weights=ngram_weights, 
                          smoothing_function=SmoothingFunction().method3)