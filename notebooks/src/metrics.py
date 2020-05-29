from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from .useful_utils import chunks
import tqdm
import numpy as np
def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')
if is_interactive():
    import tqdm.notebook as tqdm 
    
def RecipRank(q_rel_idx, scores):
    """
    q_rel_idx: int: index of the correct document matching the query in the scores array
    scores: [int]: scores given to all docs w.r.t the query
    """
    q_rel_score = scores[q_rel_idx]
    rank = sum([1 for score in scores if score>q_rel_score])
    return 1/(rank+1)

def doc_search_subtask(queries, docs, lookup, scoring_fn):
    """
    We are assuming each query at index i matches only 1 document in the collection also located at index i
    queries: [int]*n
    docs: [int]*m
    scoring_fn: fn(int,[int]*m, lookup)->[float]*m
    """
    RRs = []
    for i, query in enumerate(queries):
        scores = scoring_fn(query, docs[i])
        RRs.append(RecipRank(i, scores))
    MRR = np.average(RRs)
    return {"MRR":MRR}

def nltk_bleu(refrence, prediction):
    """
    Implementation from ReCode
    and moses multi belu script sets BLEU to 0.0 if len(toks) < 4
    refrence: [(str)]
    prediction: [(str)]
    """
    ngram_weights = [0.25] * min(4, len(refrence))
    return sentence_bleu([refrence], prediction, weights=ngram_weights, 
                          smoothing_function=SmoothingFunction().method3)

def calculate_MRR(qrel_file, run_file, lookup, ranking_fn, chunk_size=1000):
    '''
    data: [(query, true_doc)]
    ranking_fn: a function that can take in a list of [(query, doc)] and return a [score] in that same order.
    
    The index of the data in the array is going to be the id used. The first element in each subsequent 
    array corresponding to a query will be the corresponding ground truth with the rest till the hit_length be 
    randomly sampled ids from the data.
    It is assumed the data is ordered and continuous following TREC format.
    
    >>> qrel_file = "/nfs/code_search_net_archive/python/final/jsonl/valid.qrel"
    >>> run_file = "/nfs/code_search_net_archive/python/final/jsonl/valid.run"
    >>> calculate_MRR(qrel_file, run_file, full_sample_lookup, dummy_scorer)
    '''    
    # {(qid: docid)}
    qrel_lookup = {}
    num_lines = sum(1 for line in open(qrel_file))
    with open(qrel_file, "r") as q_rel_f:
        print("getting qrels")
        pbar = tqdm.tqdm(q_rel_f, total=num_lines)
        for line in pbar:
            split_line = line.strip().split()
            qrel_lookup[split_line[0]] = split_line[2]
    
    MRR_scores = []
    num_lines = sum(1 for line in open(run_file))
    with open(run_file, "r") as run_f:
        print("getting runs")
        query_chunk = []
        pbar = tqdm.tqdm(run_f, total=num_lines)
        for line in pbar:
            split_line = line.strip().split()
            query_chunk.append((split_line[0],split_line[2]))
            if len(query_chunk) >= chunk_size:
                #pocess
                idx_relevant_doc = query_chunk.index((query_chunk[0][0], qrel_lookup[query_chunk[0][0]]))
                query_doc_pairs = [(lookup[q_id]['docstring_tokens'], 
                                    lookup[d_id]['code'].replace(lookup[query_chunk[0][1]]['docstring'],"")) for q_id, d_id in query_chunk]
                scores = ranking_fn(query_doc_pairs)
                relevant_doc_score = scores.pop(idx_relevant_doc)
                rank = sum([1 for s in scores if s>relevant_doc_score])
                MRR_scores.append(1.0/(rank+1))
                query_chunk = []
        
    return np.average(MRR_scores), MRR_scores