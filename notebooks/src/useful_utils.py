from datetime import datetime
import re
import random
import os
import gzip
import json
import requests
import urllib
import jsonlines
import numpy as np
from  heapq import heappush, heappop, nsmallest
from anytree import Node, RenderTree, NodeMixin, find_by_attr
from tqdm.auto import tqdm  
from pytorch_lightning import Callback, EvalResult

class Validate_and_Save_Callback(Callback):
    def __init__(self, filepath, prefix='', validate_fn=None, interval=0.5, monitor=None, skip_epochs=[]):
        '''
        This is a callback to validate and save a model on a % interval from a pytorch lightning module.
        Given a validation function it will run and save the result while logging to the pl logger.
        If validate_fn is None, the checkpoint will be saved regardless
        
        validate_fn: fn(model) -> {'metic':0.4...}: a function that given a model calculates all the necessary scores for validation and model info
        interval: float: the fraction of the data that is passed at which an evaluation is performed
        monitor: str: the metric resulting from the validation_fn dict to check and save
        skip_epochs: [int]: a list of the epochs to skip evaluation for. Useful at the start since model is randomly initialised.
        
        >>> validate_fn = lambda model: return {'val_bleu':0.4, 'val_acc':0.35}
        >>> saving_cb = Validate_and_Save_Callback(filepath='model/path/', prefix='my_model', validate_fn=validate_fn, epoch_save_interval=0.05)
        >>> trainer = Trainer(gpus=[1], callbacks=[saving_cb])
        >>> trainer.fit(model, train_dataloader)
        '''
        self.interval = interval
        self.validate_fn = validate_fn
        self.next_save_batch_idx = 0
        self.filepath = filepath
        self.prefix = prefix
        self.skip_epochs = skip_epochs
        
        self.prev_best = 0
        self.monitor = monitor
        self.first_valid_complete = False
        
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        
    def on_batch_end(self, trainer, pl_module):
        if trainer.limit_train_batches:
            total_steps = len(trainer.train_dataloader) * trainer.limit_train_batches
        else: 
            total_steps = len(trainer.train_dataloader)
        
        if (trainer.global_step % int(total_steps*self.interval) == 0 and trainer.current_epoch not in self.skip_epochs) or not self.first_valid_complete:
            self.first_valid_complete = True
            val_dict = {}
            monitor_label = ''
            if self.validate_fn:
                val_dict = self.validate_fn(trainer.get_model())
                trainer.logger.log_metrics(val_dict, step=trainer.global_step)
                monitor_label = f'_{val_dict[self.monitor]:.5f}'
                
            filepath=os.path.join(self.filepath, f"{self.prefix}{monitor_label}_step_{trainer.global_step}.ckpt")
            if self.monitor:
                if val_dict[self.monitor] < self.prev_best:
                    return
                self.prev_best = val_dict[self.monitor]
            trainer.save_checkpoint(filepath)

def download_from_url(url, dst):
    """
    @param: url to download file
    @param: dst place to put the file
    """
    file_size = int(urllib.request.urlopen(url).info().get('Content-Length', -1))
    if os.path.exists(dst):
        first_byte = os.path.getsize(dst)
    else:
        first_byte = 0
    if first_byte >= file_size:
        return file_size
    header = {"Range": "bytes=%s-%s" % (first_byte, file_size)}
    pbar = tqdm(
        total=file_size, initial=first_byte,
        unit='B', unit_scale=True, desc=url.split('/')[-1])
    req = requests.get(url, headers=header, stream=True)
    with(open(dst, 'ab')) as f:
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                pbar.update(1024)
    pbar.close()
    return file_size
    
def dummy_scorer(query, docs):
    '''
    query: str,
    docs: [str]: docs to rank against the query
    '''
#     print(q_doc_pairs[0])
    return list(np.random.uniform(low=0.0, high=1.0, size=(len(docs),)))

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

def jsonl_dir_to_data(path):
    """
    path: /test/path/file1.jsonl
                     file2.jsonl
                     ...
    returns: [dict]: an array containing all the lines from all files in the path directory
    """
    data_samples = []
    files_list = os.listdir(path)
    files_list.sort()
    for file_name in  tqdm(files_list, desc=f"loading: {path[-10:]}"):
        file_path = os.path.join(path, file_name)
        with gzip.GzipFile(file_path, 'r') as fin:
            data = jsonlines.Reader(fin)
            for line in data.iter():
                data_samples.append(line)
    return data_samples
        
def load_CSN_data(datapath):
    """
    This function loads all the CodeSearchNet data in memory for train, valid, and test
    datapath: String: the path leading to the train, valid, test folders. Ex: "python/final/jsonl" -> /test /train /valid
    
    >>> train_data, valid_data, test_data = load_CSN_data("/nfs/code_search_net_archive/python/final/jsonl/")
    """
             
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
        
def normalize(x):
    return x/np.sum(x)


class BeamNode(NodeMixin):  # Add Node feature
    def __init__(self, ID, post_pred_state, parent=None, children=None):
        super(MyClass, self).__init__()
        self.ID = ID
        self.post_pred_state = post_pred_state
        self.parent = parent
        if children:  # set children only if given
            self.children = children
        

def beam_search(inputs_to_ids_fn, starting_state, starting_id, stop_condition_fn, beam_width=3, \
                num_states_returned=3, top_k=10, top_p=1.0, num_dist_samples=-1, temperature=1.0):
    """
    Searches a space using a beam constrrained tree to find the most likely outcomes.
    
    inputs_to_ids_fn: (state, id)->(new_state, [float]): the prediction function returning an 
                                            array of probabilities corresponding to the next valid ids
    starting_cache: an object containing the state of the prediction fn before being passed a new id
    starting_id: int: the integer used to prime the first distribution
    stop_condition_fn: (state)->bool: a function determining if the sequence should be stopped,
                                      this could include maximum length reached.
    beam_width: int: number of active beams to search at any particular point.
    
    returns: [state]: most probable terminated states
    """
    terminated_states = []
    active_states = [(0.0, starting_state, starting_id)] # we use log probabilities, and thus sum future log probs. log(p==1) = 0
    beam_tree = Node(f"Start", state=starting_state["input_ids"].cpu().tolist().copy())
    
    while len(terminated_states) < num_states_returned and len(active_states) != 0:
        p, best_state, best_next_id = heappop(active_states)
        new_state, next_id_probs = inputs_to_ids_fn(best_state, best_next_id)
        
        print("searching for", best_state["input_ids"].cpu().tolist().copy())
        parent = find_by_attr(beam_tree, name="state", value=best_state["input_ids"].cpu().tolist())
        print("found parent ",parent)
        print("adding node with state ",new_state["input_ids"].cpu().tolist().copy())
        added_node = Node(f"{best_next_id}", state=new_state["input_ids"].cpu().tolist().copy(), parent=parent)
        if stop_condition_fn(new_state):
            terminated_states.append((-p, new_state.copy()))
            continue
        
        # choose best IDs to add to tree
        next_id_probs = np.array(next_id_probs)
        scaled_logits = np.log(next_id_probs) / temperature
        exp_logits = np.exp(scaled_logits)
        next_id_probs = exp_logits / np.sum(exp_logits) # softmax
        
        sorted_ids = np.argsort(next_id_probs)[::-1]
        cumulative_probs = np.cumsum(next_id_probs[sorted_ids])
        top_p_ids = [idx for idx, cumulative_p in zip(sorted_ids,cumulative_probs) if cumulative_p<=top_p]
        top_k_ids = top_p_ids[:top_k]
        
        if num_dist_samples<1:
            sampled_ids = top_k_ids[:beam_width]
        else:
            sampled_ids = np.random.choice(top_k_ids, min(num_dist_samples, len(top_k_ids)), p=normalize(next_id_probs[top_k_ids]), replace=False)
        
        for idx in sampled_ids:
            new_prob = p-np.log(next_id_probs[idx])
            heappush(active_states, (new_prob, new_state.copy(), idx))
        
        active_states = nsmallest(beam_width, active_states)
        
    for pre, fill, node in RenderTree(beam_tree):
        print("%s%s" % (pre, node.name))  
    
    return terminated_states