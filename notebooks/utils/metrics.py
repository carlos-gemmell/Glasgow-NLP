from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

def nltk_bleu(refrence, prediction):
    """
    Implementation from ReCode
    and moses multi belu script sets BLEU to 0.0 if len(toks) < 4
    """
    ngram_weights = [0.25] * min(4, len(refrence))
    return sentence_bleu([refrence], prediction, weights=ngram_weights, 
                          smoothing_function=SmoothingFunction().method3)