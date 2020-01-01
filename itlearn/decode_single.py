import ipdb
import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from data import NormalField, NormalTranslationDataset, TripleTranslationDataset
from utils import write_tb
from metrics import Metrics, Best
from misc.bleu import computeBLEU, compute_bp, print_bleu
from pathlib import Path

def valid_model(args, model, dev_it, dev_metrics, decode_method, beam_width=5, test_set="valid"):
    with torch.no_grad():
        model.eval()
        src_corpus, trg_corpus, hyp_corpus = [], [], []

        for j, dev_batch in enumerate(dev_it):
            batch_size = len(dev_batch)
            src, src_len = dev_batch.src
            trg, trg_len = dev_batch.trg

            logits, _ = model(src[:,1:], src_len-1, trg[:,:-1])
            nll = F.cross_entropy(logits, trg[:,1:].contiguous().view(-1), size_average=True, ignore_index=0, reduce=True)
            num_trg = (trg[:,1:] != 0).sum().item()

            dev_metrics.accumulate(num_trg, nll.item())

            hyp = model.decode(src, src_len, decode_method, beam_width)
            src_corpus.extend( args.src.reverse( src ) )
            trg_corpus.extend( args.trg.reverse( trg ) )
            hyp_corpus.extend( args.trg.reverse( hyp ) )

        bleu = computeBLEU(hyp_corpus, trg_corpus, corpus=True)
        args.logger.info(dev_metrics)
        args.logger.info("{} {} : {}".format(test_set, decode_method, print_bleu(bleu)))
        args.logger.info('model:' + args.prefix + args.id_str)

        if not args.debug:
            src = Path( args.decoding_path ) / args.id_str / "src"
            trg = Path( args.decoding_path ) / args.id_str / "trg"
            hyp = Path( args.decoding_path ) / args.id_str / "hyp"
            src.write_text("\n".join(src_corpus), encoding="utf-8")
            trg.write_text("\n".join(trg_corpus), encoding="utf-8")
            hyp.write_text("\n".join(hyp_corpus), encoding="utf-8")

    return bleu

def decode_model(args, model, iterators):
    (dev_it) = iterators

    if args.gpu > -1 and torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = lambda storage, loc: storage

    pretrained = torch.load( "{}_best.pt".format(args.load_from), map_location )
    model.load_state_dict( pretrained )
    args.logger.info("Pretrained model loaded.")

    if not args.debug:
        decoding_path = Path(args.decoding_path + args.id_str)
        decoding_path.mkdir(parents=True, exist_ok=True)
        from tensorboardX import SummaryWriter
        writer = SummaryWriter( args.event_path + args.id_str)

    dev_metrics = Metrics('dev_loss', 'nll', data_type = "avg")
    dev_bleu = valid_model(args, model, dev_it, dev_metrics, args.decode_method)
