import argparse
from pathlib import Path
import numpy as np

from misc.bleu import computeBLEU, compute_bp, print_bleu, sentence_bleu

def get_sentence_bleus(hyp_path, ref_path, gram=2):
    refs = ref_path.read_text().split('\n')
    hyps = hyp_path.read_text().split('\n')
    if gram == 3:
        weights = (0.333, 0.333, 0.333)
    elif gram == 2:
        weights = (0.5, 0.5)
    elif gram == 1:
        weights = (1.0, 0.0)
    bleus = [sentence_bleu([ref.split()], hyp.split(), weights=(1.0, 0.0))[0] for (ref, hyp) in zip(refs, hyps)]
    return bleus, refs, hyps

def rank_helper(path, by, iters, gram=2):
    en_bleus, en_refs, en_hyps = get_sentence_bleus( path / "en_hyp_{}".format(iters), path / "en_ref", gram)
    de_bleus, de_refs, de_hyps = get_sentence_bleus( path / "de_hyp_{}".format(iters), path / "de_ref", 2)
    diffs = [de_bleu - 1.5 * en_bleu for (de_bleu, en_bleu) in zip(de_bleus, en_bleus)]
    if by == "diff":
        indices = np.argsort(np.array(diffs))[::-1] # diff bigger the better
    elif by == "en":
        indices = np.argsort(np.array(en_bleus)) # EN smaller the better
    return indices, diffs, (en_refs, en_hyps), (de_refs, de_hyps), (en_bleus, de_bleus)

def find(path, words="punk", cands="kid child children toddler baby infant", how="hyp2ref", more=False):
    path = "/checkpoint/jasonleeinf/groundcomms/decoding_180609/multi30k/" + path
    #path = "/checkpoint/jasonleeinf/groundcomms/decoding_180606_3/multi30k/" + path
    p = Path(path)
    _, _, (en_refs, en_hyps), (de_refs, de_hyps), (en_bleus, de_bleus) = rank_helper(p, "diff")
    indices = []
    words = words.split()
    if how == "hyp2ref":
        _from, _to = en_hyps, en_refs
    elif how == "ref2hyp":
        _from, _to = en_refs, en_hyps

    for idx, en_hyp in enumerate(_from):
        for word in words:
            if word in en_hyp.split():
                indices.append(idx)
                break

    for idx in indices:
        print ("EN ref |", en_refs[idx])
        print ("EN hyp |", en_hyps[idx])
        print ("")

    if more:
        cnt = 0
        cands = cands.split()
        for idx in indices:
            for cand in cands:
                if cand in _to[idx]:
                    cnt += 1
                    break
        print ("{}/{}".format(cnt, len(indices)))

def rank(exp, model_id, iters, by="diff", howmany=50, bottom=False, gram=1):
    path = "/checkpoint/jasonleeinf/groundcomms/decoding/{}/{}".format(exp, model_id)
    p = Path(path)
    indices, diffs, (en_refs, en_hyps), (de_refs, de_hyps), (en_bleus, de_bleus) = rank_helper(p, by, iters, gram)
    for ii, index in enumerate(indices[:howmany]):
        print ("TOP {}".format(ii+1))
        print ("EN ref |", en_refs[index])
        print ("EN hyp |", en_hyps[index])
        print ("DE ref |", de_refs[index])
        print ("DE hyp |", de_hyps[index])
        print ("EN BLEU={:.2f},   DE BLEU={:.2f},   DIFF={:.2f}".format(en_bleus[index], de_bleus[index], diffs[index]))
        print ("")

    if bottom:
        for ii, index in enumerate(indices[-howmany:]):
            print ("BOTTOM {}".format(howmany-ii))
            print ("EN ref |", en_refs[index])
            print ("EN hyp |", en_hyps[index])
            print ("DE ref |", de_refs[index])
            print ("DE hyp |", de_hyps[index])
            print ("EN BLEU={:.2f},   DE BLEU={:.2f},   DIFF={:.2f}".format(en_bleus[index], de_bleus[index], diffs[index]))
            print ("")

"""
parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str)
args = parser.parse_args()

p = Path(args.path)
subdirs = [x for x in p.iterdir() if x.is_dir()]

for subdir in subdirs:
    fullpath = p / subdir
    indices, diffs, (en_refs, en_hyps), (de_refs, de_hyps) = rank_helper(fullpath)
"""
