from src.retrieval import PyLuceneRetriever, OracleBLEURetriever

def retrieval_output_nudging_creator(oracle=False, relevance_interpol=0.0005, k_docs=10, k_words=10, peak_scaling_factor=40.0, \
                                     num_stop_words=10, verbose=False):
    src_train_samples = [src for src, tgt in train_samples]
    tgt_train_samples = [tgt for src, tgt in train_samples]
    if oracle:
        retriever = OracleBLEURetriever(ids_to_keep=k_docs)
        retriever.add_multiple_docs(train_samples)
    else:
        retriever = PyLuceneRetriever()
        retriever.add_multiple_docs(src_train_samples)
    
    counts = Counter(string_split_v3(" ".join(tgt_train_samples))).most_common(num_stop_words)
    stop_words = [x[0] for x in counts]
    
    def nudge_fn(last_token_log_probs, single_decoder_input, batch_encoder_ids, batch_decoder_truth_ids, OOVs):
        OOVs = OOVs.cpu().tolist()
        src_sent = vocab.decode_input(batch_encoder_ids, OOVs, copy_marker="")
        tgt_sent = vocab.decode_output(batch_decoder_truth_ids, OOVs, copy_marker="")
        current_pred = vocab.decode_output(single_decoder_input, OOVs, copy_marker="")
        top_5_ids = torch.argsort(last_token_log_probs.cpu(), descending=True)[:5]
        top_5_words = [vocab.decode_output([idx], OOVs, copy_marker="") for idx in top_5_ids]
        if verbose:
            print("## DECODE STEP ##")
            print(f"SRC input:      {src_sent}")
            print(f"TGT truth:      {tgt_sent}")
            print(f"decoded so far: {current_pred}")
            print(f"top words     : {' | '.join(top_5_words)}")
            print()
        if oracle:
            doc_ranking = retriever.search(src_sent, tgt_sent, max_retrieved_docs=k_docs)
        else:
            doc_ranking = retriever.search(src_sent, max_retrieved_docs=k_docs)
            
        retrieved_samples = [(tgt_train_samples[doc_id], score) for doc_id, score in doc_ranking]
        scoring_dict = {}
        for sample, score in retrieved_samples:
            if verbose:
                print(f"DOC: {sample}")
            sample_toks = string_split_v3(sample)
            for tok in sample_toks:
                if tok in scoring_dict:
                    scoring_dict[tok] += (peak_scaling_factor * score)/len(sample_toks)
                else:
                    scoring_dict[tok] = (peak_scaling_factor * score)/len(sample_toks)
        top_retrieved_words = [tok for tok in sorted(scoring_dict.items(), key=lambda item: -item[1]) if tok[0] not in stop_words][:k_words]
        if verbose:
            print(f"RETRIEVAL top words: {[tok for tok, score in top_retrieved_words]}")
            print()
            print()
        top_retrieved_ids = [(vocab.encode_output(tok, OOVs)[0], score) for tok, score in top_retrieved_words]
        top_retrieved_ids = [(i, s) for i, s in top_retrieved_ids if i != vocab.UNK]
        
        relevance_vector = torch.zeros_like(last_token_log_probs).fill_(-5000.0)
        for idx, score in top_retrieved_ids:
            if idx not in single_decoder_input:
                relevance_vector[idx] = score
        relevance_vector.softmax(-1)
        
        if top_5_ids[0] == vocab.EOS:
            new_probs = last_token_log_probs
        else:
            new_probs = (1-relevance_interpol) * last_token_log_probs + relevance_interpol * relevance_vector
        
        new_top_pred = torch.argmax(new_probs)
        if verbose:
            if top_5_ids[0] != new_top_pred:
                print("Relevance impact:")
                print(f"SRC input:      {src_sent}")
                print(f"TGT truth:      {tgt_sent}")
                print(f"decoded so far: {current_pred}")
                print(f"RETRIEVAL top words: {[tok for tok, score in top_retrieved_words][:]}")
                print(f"Prerdicted {vocab.decode_output([new_top_pred], OOVs)} over {top_5_words[0]}")
                print()
            
        return new_probs
    
    return nudge_fn