import torch

from utils.bleu import computeBLEU


def valid_model(args, model, dev_it):
    with torch.no_grad():
        unbpe = True
        model.eval()
        fr_corpus, en_corpus, de_corpus = [], [], []
        en_hyps, de_hyp = [ [] for ii in range(5)], []

        total_msg, total_ref = 0, 0
        for j, dev_batch in enumerate(dev_it):
            fr_corpus.extend( args.FR.reverse(dev_batch.fr[0], unbpe=unbpe) )
            en_corpus.extend( args.EN.reverse(dev_batch.en[0], unbpe=unbpe) )
            de_corpus.extend( args.DE.reverse(dev_batch.de[0], unbpe=unbpe) )

            en_msgs, de_msg = model.multi_decode(dev_batch)
            [ en_hyp.extend( args.EN.reverse(en_msg, unbpe=unbpe) ) for en_hyp, en_msg in zip(en_hyps, en_msgs)]
            de_hyp.extend( args.DE.reverse(de_msg, unbpe=unbpe) )

        bleu_ens = [ computeBLEU(en_hyp, en_corpus, corpus=True) for en_hyp in en_hyps ]
        bleu_de = computeBLEU(de_hyp, de_corpus, corpus=True)
        for bleu_en in bleu_ens:
            print ("EN", bleu_en)
        print ("DE", bleu_de)

def decode_model(args, model, iterators):
    (dev_it) = iterators
    R = valid_model(args, model, dev_it)
