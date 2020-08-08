class Run_File_Searcher():
    def __init__(self, run_file, **kwargs):
        '''
        >>> GPT2_rewriter_BERT_rerank_model = Run_File_Reranker("saved_models/CAsT_y1_pgbert.run")
        '''
        self.query_doc_mapping = {}
        with open(run_file, "r") as run_f:
            for line in run_f:
                q_id, _, doc_id, rank_str, score_str, name = line.split(" ")
                score = float(score_str)
                
                if q_id not in self.query_doc_mapping:
                    self.query_doc_mapping[q_id] = []
                    
                self.query_doc_mapping[q_id].append((doc_id, score))
    
    def predict(self, q_id, hits=10):
        return self.query_doc_mapping[q_id][:hits]
    
    def batch_predict(self, q_ids, **kwargs):
        return [predict(q_id, **kwargs) for q_id in q_ids]