import argparse
from pathlib import Path
import numpy as np

from misc.bleu import sentence_bleu

def get_sentence_bleus(hyp_path, ref_path, gram, dobp):
    refs = ref_path.read_text().split('\n')
    hyps = hyp_path.read_text().split('\n')
    if gram == 1:
        weights = [1/1]
    elif gram == 2:
        weights = (1/2, 1.2)
    elif gram == 3:
        weights = (1/3, 1/3, 1/3)
    elif gram == 4:
        weights = (1/4, 1/4, 1/4, 1/4)

    bleus = []
    for (ref, hyp) in zip(refs, hyps):
        final_bleu, p_n_, bp, ref_lengths, hyp_lengths = sentence_bleu([ref.split()], hyp.split(), weights=weights)
        if dobp:
            bleu = final_bleu
        else:
            if bp != 0:
                bleu = final_bleu / bp

        bleus.append( bleu )

    return bleus, refs, hyps

def rank_helper(path, by, gram, dobp):
    en_bleus, en_refs, en_hyps = get_sentence_bleus( path / "en_hyp", path / "en_ref", gram=gram, dobp=dobp)
    de_bleus, de_refs, de_hyps = get_sentence_bleus( path / "de_hyp", path / "de_ref", gram=gram, dobp=dobp)
    diffs = [de_bleu - en_bleu for (de_bleu, en_bleu) in zip(de_bleus, en_bleus)]
    if by == "diff":
        indices = np.argsort(np.array(diffs))[::-1] # diff bigger the better
    elif by == "en":
        indices = np.argsort(np.array(en_bleus)) # EN smaller the better
    return indices, diffs, (en_refs, en_hyps), (de_refs, de_hyps), (en_bleus, de_bleus)

def rank(cpt, alpha, lr, by="diff", gram=2, dobp=True, howmany=20, bottom=False):
    path = Path("/checkpoint/jasonleeinf/groundcomms/decoding_180612_cpt_alpha/multi30k/")
    subdirs = []
    for subdir in path.iterdir():
        subdir_ = str(subdir)
        if "cpt{}_".format(cpt) in subdir_ and \
           "msg{}x_".format(alpha) in subdir_ and \
           "lr{:.0e}_".format(lr) in subdir_:
            subdirs.append(subdir_)
            print (subdir_)

    assert len(subdirs) == 1
    p = path / subdirs[0]
    print ("Extracting decodings from : \n {}".format(str(p)))
    indices, diffs, (en_refs, en_hyps), (de_refs, de_hyps), (en_bleus, de_bleus) = rank_helper(p, by, gram, dobp)
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
    print ("Extracting decodings from : \n {}".format(str(p)))

# rank(1000, 1.0, 1e-4, gram=1, dobp=True, howmany=100)
def find(cpt, alpha, lr, words="thumb", cands="children little small baby girl child", how="hyp2ref", more=False):
    path = Path("/checkpoint/jasonleeinf/groundcomms/decoding_180612_cpt_alpha/multi30k/")
    subdirs = []
    for subdir in path.iterdir():
        subdir_ = str(subdir)
        if "cpt{}_".format(cpt) in subdir_ and \
           "msg{}x_".format(alpha) in subdir_ and \
           "lr{:.0e}_".format(lr) in subdir_:
            subdirs.append(subdir_)
            print (subdir_)

    assert len(subdirs) == 1
    p = path / subdirs[0]

    _, _, (en_refs, en_hyps), (de_refs, de_hyps), (en_bleus, de_bleus) = rank_helper(p, "diff", gram=2, dobp=True)
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

