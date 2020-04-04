import math
import torch
import random
import numpy as np
import _pickle as pickle
import revtok
import os
from itertools import groupby
import getpass
from collections import Counter

from torchtext import data, datasets
from nltk.translate.gleu_score import sentence_gleu, corpus_gleu
from nltk.translate.bleu_score import closest_ref_length, brevity_penalty, modified_precision, SmoothingFunction
from contextlib import ExitStack
from collections import OrderedDict
import fractions

try:
    fractions.Fraction(0, 1000, _normalize=False)
    from fractions import Fraction
except TypeError:
    from nltk.compat import Fraction

def sentence_bleu(references, hypothesis, weights=(0.25, 0.25, 0.25, 0.25),
                  smoothing_function=None, auto_reweigh=False,
                  emulate_multibleu=False):

    return corpus_bleu([references], [hypothesis],
                        weights, smoothing_function, auto_reweigh,
                        emulate_multibleu)

def corpus_bleu(list_of_references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=None, auto_reweigh=False,
                emulate_multibleu=False):
    p_numerators = Counter() # Key = ngram order, and value = no. of ngram matches.
    p_denominators = Counter() # Key = ngram order, and value = no. of ngram in ref.
    hyp_lengths, ref_lengths = 0, 0

    if len(list_of_references) != len(hypotheses):
        print ("The number of hypotheses and their reference(s) should be the same")
        return (0, *(0, 0, 0, 0), 0, 0, 0)

    # Iterate through each hypothesis and their corresponding references.
    for references, hypothesis in zip(list_of_references, hypotheses):
        # For each order of ngram, calculate the numerator and
        # denominator for the corpus-level modified precision.
        for i, _ in enumerate(weights, start=1):
            p_i = modified_precision(references, hypothesis, i)
            p_numerators[i] += p_i.numerator
            p_denominators[i] += p_i.denominator

        # Calculate the hypothesis length and the closest reference length.
        # Adds them to the corpus-level hypothesis and reference counts.
        hyp_len =  len(hypothesis)
        hyp_lengths += hyp_len
        ref_lengths += closest_ref_length(references, hyp_len)

    # Calculate corpus-level brevity penalty.
    bp = brevity_penalty(ref_lengths, hyp_lengths)

    # Uniformly re-weighting based on maximum hypothesis lengths if largest
    # order of n-grams < 4 and weights is set at default.
    if auto_reweigh:
        if hyp_lengths < 4 and weights == (0.25, 0.25, 0.25, 0.25):
            weights = ( 1 / hyp_lengths ,) * hyp_lengths

    # Collects the various precision values for the different ngram orders.
    p_n = [Fraction(p_numerators[i], p_denominators[i], _normalize=False)
           for i, _ in enumerate(weights, start=1)]

    p_n_ = [xx.numerator / xx.denominator * 100 for xx in p_n]

    # Returns 0 if there's no matching n-grams
    # We only need to check for p_numerators[1] == 0, since if there's
    # no unigrams, there won't be any higher order ngrams.
    if p_numerators[1] == 0:
        return (0, *(0, 0, 0, 0), 0, 0, 0)

    # If there's no smoothing, set use method0 from SmoothinFunction class.
    if not smoothing_function:
        smoothing_function = SmoothingFunction().method0
    # Smoothen the modified precision.
    # Note: smoothing_function() may convert values into floats;
    #       it tries to retain the Fraction object as much as the
    #       smoothing method allows.
    p_n = smoothing_function(p_n, references=references, hypothesis=hypothesis,
                             hyp_len=hyp_len, emulate_multibleu=emulate_multibleu)
    s = (w * math.log(p_i) for i, (w, p_i) in enumerate(zip(weights, p_n)))
    s =  bp * math.exp(math.fsum(s)) * 100
    final_bleu = round(s, 4) if emulate_multibleu else s
    return (final_bleu, *p_n_, bp, ref_lengths, hyp_lengths)

def computeBLEU(outputs, targets, corpus=False, tokenizer=None):
    if tokenizer is None:
        tokenizer = lambda x: x.replace('@@ ', '').split()

    outputs = [tokenizer(o) for o in outputs]
    targets = [tokenizer(t) for t in targets]

    if corpus:
        return corpus_bleu([[t] for t in targets], [o for o in outputs], emulate_multibleu=True)
    else:
        return [sentence_bleu([t],  o)[0] for o, t in zip(outputs, targets)]

def compute_bp(hypotheses, list_of_references):
    hyp_lengths, ref_lengths = 0, 0
    for references, hypothesis in zip(list_of_references, hypotheses):
        hyp_len =  len(hypothesis)
        hyp_lengths += hyp_len
        ref_lengths += closest_ref_length(references, hyp_len)

    # Calculate corpus-level brevity penalty.
    bp = brevity_penalty(ref_lengths, hyp_lengths)
    return bp

def computeGroupBLEU(outputs, targets, tokenizer=None, bra=10, maxmaxlen=80):
    if tokenizer is None:
        tokenizer = revtok.tokenize

    outputs = [tokenizer(o) for o in outputs]
    targets = [tokenizer(t) for t in targets]
    maxlens = max([len(t) for t in targets])
    print(maxlens)
    maxlens = min([maxlens, maxmaxlen])
    nums = int(np.ceil(maxlens / bra))
    outputs_buckets = [[] for _ in range(nums)]
    targets_buckets = [[] for _ in range(nums)]
    for o, t in zip(outputs, targets):
        idx = len(o) // bra
        if idx >= len(outputs_buckets):
            idx = -1
        outputs_buckets[idx] += [o]
        targets_buckets[idx] += [t]

    for k in range(nums):
        print(corpus_bleu([[t] for t in targets_buckets[k]], [o for o in outputs_buckets[k]], emulate_multibleu=True))

def print_bleu(bleu_output, verbose=True):
    (final_bleu, *prec, bp, ref_lengths, hyp_lengths) = bleu_output
    ratio = 0 if ref_lengths == 0 else hyp_lengths/ref_lengths
    if verbose:
        return "BLEU = {:.2f}, {:.1f}/{:.1f}/{:.1f}/{:.1f} (BP={:.3f}, ratio={:.3f}, hyp_len={}, ref_len={})".format(
            final_bleu, prec[0], prec[1], prec[2], prec[3], bp, ratio, hyp_lengths, ref_lengths
        )
    else:
        return "BLEU = {:.2f}, {:.1f}/{:.1f}/{:.1f}/{:.1f} (BP={:.3f}, ratio={:.3f})".format(
            final_bleu, prec[0], prec[1], prec[2], prec[3], bp, ratio
        )
