from pyserini.search import SimpleSearcher

class BM25_Ranker():
    def __init__(self, index_dir="/nfs/phd_by_carlos/notebooks/datasets/TREC_CAsT/CAsT_collection_with_meta.index", **kwargs):
        self.searcher = SimpleSearcher(index_dir)
        
    def predict(self, query, hits=10, **kwargs):
        search_results = self.searcher.search(query, k=hits)
        len_res = min(hits, len(search_results))
        results = [(search_results[i].docid, search_results[i].score) for i in range(len_res)]
        return results