# https://github.com/golsun/NLP-tools/blob/master/metrics.py#L158

import nltk.translate.bleu_score as bleu
import numpy as np
import re

from load_data import process_data
from nltk.translate.bleu_score import SmoothingFunction

smoothie = SmoothingFunction().method4


def bleu_score(path_refs, path_hyp, n_lines):
    refs_ls, bleu_ls = [], []
    for r, path in enumerate(path_refs):
        refs_ls.append([])
        refs_ls[r] = [line.strip('\n') for line in open(path, encoding='utf-8')]
    hyps = [line.strip('\n') for line in open(path_hyp, encoding='utf-8')]
    
    hyps = [process_data(text) for text in hyps]
    hyps = [re.sub(r'([,!.?])', r'', text) for text in hyps]
    
    ngram_weight = dict()
    for ngram in range(1, 5):
        ngram_weight[ngram] = [1./ngram] * ngram
    
    for ngram in ngram_weight:
        _bleu = []
        for idx in range(len(hyps)):
            multi_ref = [refs_ls[r][idx].split() for r in range(len(refs_ls))]
            _bleu.append(bleu.sentence_bleu(multi_ref, hyps[idx].split(),
                                            weights=ngram_weight[ngram],
                                            smoothing_function=smoothie))
        bleu_ls.append(np.mean(_bleu))
    return bleu_ls