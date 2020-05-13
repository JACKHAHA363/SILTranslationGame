from torch import nn as nn
import torch
from pathlib import Path
import math
import numpy as np

from itlearn.utils.metrics import Metrics, Best
from itlearn.utils.misc import write_tb, plot_grad
from itlearn.finetune.agents_utils import eval_model, valid_model


def joint_loop(args, model, train_it, dev_it, extra_input, loss_cos, loss_names, monitor_names):
    # Prepare writer
    if not args.debug:
        decoding_path = Path(args.decoding_path + args.id_str)
        decoding_path.mkdir(parents=True, exist_ok=True)
        from tensorboardX import SummaryWriter
        writer = SummaryWriter( args.event_path + args.id_str)

    # Prepare opt
    if args.fix_fr2en:
        args.logger.info('Fix Fr En')
        params = [p for p in model.en_de.parameters() if p.requires_grad]
    else:
        params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == 'Adam':
        opt = torch.optim.Adam(params, betas=(0.9, 0.98), eps=1e-9, lr=args.lr)
    else:
        raise NotImplementedError

    train_metrics = Metrics('train_loss', *loss_names, *monitor_names, data_type="avg")
    best = Best(max, 'de_bleu', 'en_bleu', 'iters', model=model, opt=opt, path=args.model_path + args.id_str,
                gpu=args.gpu, debug=args.debug)

    fr_en_it, en_de_it = None, None
    s2p_steps = args.__dict__.get('s2p_steps', args.max_training_steps)
    if hasattr(args, 's2p_freq') and args.s2p_freq > 0:
        args.logger.info('Perform S2P at every {} steps'.format(args.s2p_freq))
        fr_en_it, en_de_it = extra_input['s2p_its']['fr_en'], extra_input['s2p_its']['en_de']
        fr_en_it = iter(fr_en_it)
        en_de_it = iter(en_de_it)

    for iters, train_batch in enumerate(train_it):
        if iters >= args.max_training_steps:
            args.logger.info('stopping training after {} training steps'.format(args.max_training_steps))
            break

       # Add S2P Grad
        if args.s2p_freq > 0 and iters % args.s2p_freq == 0 and iters <= s2p_steps:
            fr_en_batch = fr_en_it.__next__()
            en_de_batch = en_de_it.__next__()

        if iters % args.eval_every == 0:
            args.logger.info("update {}".format(iters))


def s2p_batch(fr_en_it, en_de_it, agents):
    fr_en_batch = fr_en_it.__next__()
    en_de_batch = en_de_it.__next__()
    fr_en_loss = _get_nll(agents.fr_en,
                          fr_en_batch.src[0],
                          fr_en_batch.src[1],
                          fr_en_batch.trg[0])
    en_de_loss = _get_nll(agents.en_de,
                          en_de_batch.src[0],
                          en_de_batch.src[1],
                          en_de_batch.trg[0])
    return fr_en_loss, en_de_loss


def _get_nll(model, src, src_len, trg):
    logits, _ = model(src[:, 1:], src_len - 1, trg[:, :-1])
    loss = torch.nn.functional.cross_entropy(logits, trg[:, 1:].contiguous().view(-1),
                                             reduction='mean', ignore_index=0)
    return loss
