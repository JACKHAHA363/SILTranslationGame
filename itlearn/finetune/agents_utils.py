import torch
import torch.nn.functional as F

from itlearn.utils.metrics import Metrics
from itlearn.utils.bleu import computeBLEU


def supervise_evaluate_loop(agent, dev_it, dataset='iwslt', pair='fr_en'):
    dev_metrics = Metrics('s2p_dev', *['nll'])
    with torch.no_grad():
        agent.eval()
        trg_corpus, hyp_corpus = [], []

        for j, dev_batch in enumerate(dev_it):
            if dataset == "iwslt" or dataset == 'iwslt_small':
                src, src_len = dev_batch.src
                trg, trg_len = dev_batch.trg
                trg_field = dev_batch.dataset.fields['trg']
            elif dataset == "multi30k":
                src_lang, trg_lang = pair.split("_")
                src, src_len = dev_batch.__dict__[src_lang]
                trg, trg_len = dev_batch.__dict__[trg_lang]
                trg_field = dev_batch.dataset.fields[trg_lang]
            logits, _ = agent(src[:, 1:], src_len - 1, trg[:, :-1])
            nll = F.cross_entropy(logits, trg[:, 1:].contiguous().view(-1), reduction='mean',
                                  ignore_index=0)
            num_trg = (trg_len - 1).sum().item()
            dev_metrics.accumulate(num_trg, **{'nll': nll})
            hyp = agent.decode(src, src_len, 'greedy', 0)
            trg_corpus.extend(trg_field.reverse(trg, unbpe=True))
            hyp_corpus.extend(trg_field.reverse(hyp, unbpe=True))
        bleu = computeBLEU(hyp_corpus, trg_corpus, corpus=True)
    return dev_metrics, bleu
