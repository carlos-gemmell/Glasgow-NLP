from datetime import datetime
from torchtext.data import Field, BucketIterator
import re
import random
import os
import gzip
import json
import jsonlines
import tqdm
def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')
if is_interactive():
    import tqdm.notebook as tqdm 

def batch_filter_ids(batch_list, unwanted_ids):
    return [[id for id in l if id not in unwanted_ids] for l in batch_list]

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
            if os.path.exists(filename):
                action = 'a' # append if already exists
            else:
                action = 'w' 
            with open(filename,action, encoding="utf-8") as outputfile:
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

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        
        
def load_CSN_data(datapath):
    """
    This function loads all the CodeSearchNet data in memory for train, valid, and test
    datapath: String: the path leading to the train, valid, test folders. Ex: "python/final/jsonl" -> /test /train /valid
    
    >>> train_data, valid_data, test_data = load_CSN_data("/nfs/code_search_net_archive/python/final/jsonl/")
    """
    def jsonl_dir_to_data(path):
        data_samples = []
        files_list = os.listdir(path)
        files_list.sort()
        for file_name in  tqdm.tqdm(files_list):
            file_path = os.path.join(path, file_name)
            with gzip.GzipFile(file_path, 'r') as fin:
                data = jsonlines.Reader(fin)
                for line in data.iter():
                    data_samples.append(line)
        return data_samples
             
    train_data = jsonl_dir_to_data(os.path.join(datapath, "train"))
    valid_data = jsonl_dir_to_data(os.path.join(datapath, "valid"))
    test_data = jsonl_dir_to_data(os.path.join(datapath, "test"))
    
    return train_data, valid_data, test_data


def create_qrel_file(data, target_file_path):
    '''
    data: [{k:v,...}], contains 
    
    >>> train_data, valid_data, test_data = load_CSN_data("/nfs/code_search_net_archive/python/final/jsonl/")
    >>> create_qrel_file(valid_data, "/nfs/code_search_net_archive/python/final/jsonl/valid.qrel")
    '''
    with open(target_file_path, "w") as qrel_f:
        for sample in data:
            qrel_f.write(f"{sample['url'].replace(' ','%20')} 0 {sample['url'].replace(' ','%20')} 1\n")
            
            
def create_run_file(data, target_file_path, hit_length=1000):
    '''
    data: [{k:v,...}], contains 
    
    >>> train_data, valid_data, test_data = load_CSN_data("/nfs/code_search_net_archive/python/final/jsonl/")
    >>> create_run_file(valid_data, "/nfs/code_search_net_archive/python/final/jsonl/valid.run")
    '''
    assert len(data) >= hit_length
    
    run_array = []
    print("Creating runs")
    for i in tqdm.tqdm(range(len(data))):
        arr = list(range(len(data)))
        arr.pop(i)
        distractor_doc_indexes = random.sample(arr, hit_length-1)
        run_array.append([i]+distractor_doc_indexes)
    
    with open(target_file_path, "w") as run_f:
        for i in tqdm.tqdm(range(len(run_array))):
            url_list = [data[j]['url'].replace(" ","%20") for j in run_array[i]]
            full_str=""
            for k, url in enumerate(url_list):
                full_str += f"{data[i]['url'].replace(' ','%20')} Q0 {url} {k} 0.99 first_sample\n"
            run_f.write(full_str)
            

def dummy_scorer(run):
    '''
    run: [(query, doc)]
    '''
    return list(np.random.uniform(low=0.0, high=1.0, size=(len(run),)))



def read_in_chunks(file_object, chunk_size=1024):
    """Lazy function (generator) to read a file piece by piece.
    Default chunk size: 1k.
    
    >>> with open('really_big_file.dat') as f:
    >>> for piece in read_in_chunks(f):
    >>>     process_data(piece)
    """
    while True:
        data = file_object.read(chunk_size)
        if not data:
            break
        yield data