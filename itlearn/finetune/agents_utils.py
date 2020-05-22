from pathlib import Path

import torch
import torch.nn.functional as F

from itlearn.utils.metrics import Metrics
from itlearn.utils.bleu import computeBLEU


def supervise_evaluate_loop(agent, dev_it, dataset='iwslt', pair='fr_en'):
    dev_metrics = Metrics('s2p_dev', ['nll'])
    with torch.no_grad():
        agent.eval()
        src_corpus, trg_corpus, hyp_corpus = [], [], []

        for j, dev_batch in enumerate(dev_it):
            if dataset == "iwslt" or dataset == 'iwslt_small':
                src, src_len = dev_batch.src
                trg, trg_len = dev_batch.trg
            elif dataset == "multi30k":
                src_lang, trg_lang = pair.split("_")
                src, src_len = dev_batch.__dict__[src_lang]
                trg, trg_len = dev_batch.__dict__[trg_lang]
            logits, _ = agent(src[:, 1:], src_len - 1, trg[:, :-1])
            nll = F.cross_entropy(logits, trg[:, 1:].contiguous().view(-1), reduction='mean',
                                  ignore_index=0)
            num_trg = (trg_len - 1).sum().item()
            dev_metrics.accumulate(num_trg, nll.item())
            hyp = agent.decode(src, src_len, 'argmax', 0)
            src_corpus.extend(agent.src.reverse(src))
            trg_corpus.extend(agent.trg.reverse(trg))
            hyp_corpus.extend(agent.trg.reverse(hyp))
        bleu = computeBLEU(hyp_corpus, trg_corpus, corpus=True)
    return dev_metrics, bleu
