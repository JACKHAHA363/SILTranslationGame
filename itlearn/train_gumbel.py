"""
Gumbel training of joint
"""
import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from data import NormalField, NormalTranslationDataset, TripleTranslationDataset
from utils import token_analysis, get_counts, write_tb, plot_grad, cuda
from metrics import Metrics, Best
from misc.bleu import computeBLEU, compute_bp, print_bleu

from pathlib import Path


def eval_model(args, model, dev_it, monitor_names, iters, extra_input):
    """ Use greedy decoding and check scores like BLEU, language model and grounding """
    eval_metrics = Metrics('dev_loss', *monitor_names, data_type="avg")
    eval_metrics.reset()
    with torch.no_grad():
        unbpe = True
        model.eval()
        fr_corpus, en_corpus, de_corpus = [], [], []
        en_hyp, de_hyp = [], []

        for j, dev_batch in enumerate(dev_it):
            fr_corpus.extend(args.FR.reverse(dev_batch.fr[0], unbpe=unbpe))
            en_corpus.extend(args.EN.reverse(dev_batch.en[0], unbpe=unbpe))
            de_corpus.extend(args.DE.reverse(dev_batch.de[0], unbpe=unbpe))

            en_msg, de_msg, en_msg_len, _ = model.decode(dev_batch)
            en_hyp.extend(args.EN.reverse(en_msg, unbpe=unbpe))
            de_hyp.extend(args.DE.reverse(de_msg, unbpe=unbpe))
            results, _ = model.eval_fr_en_stats(en_msg, en_msg_len, dev_batch,
                                                en_lm=extra_input["en_lm"],
                                                all_img=extra_input["img"]['multi30k'][1],
                                                ranker=extra_input["ranker"])
            if len(monitor_names) > 0:
                eval_metrics.accumulate(len(dev_batch), *[results[k].item() for k in monitor_names])

        bleu_en = computeBLEU(en_hyp, en_corpus, corpus=True)
        bleu_de = computeBLEU(de_hyp, de_corpus, corpus=True)
        args.logger.info(eval_metrics)
        args.logger.info("Fr-En {} : {}".format('valid', print_bleu(bleu_en)))
        args.logger.info("En-De {} : {}".format('valid', print_bleu(bleu_de)))

        if not args.debug:
            dest_folders = [Path(args.decoding_path) / args.id_str / name for name in
                            ["en_ref", "de_ref", "fr_ref", "de_hyp_{}".format(iters), "en_hyp_{}".format(iters)]]
            [dest.write_text("\n".join(string), encoding="utf-8")
             for (dest, string) in zip(dest_folders, [en_corpus, de_corpus, fr_corpus, de_hyp, en_hyp])]
        return eval_metrics, bleu_en, bleu_de


def valid_model(model, dev_it, loss_names, monitor_names, extra_input):
    """ Run reinforce on validation and record stats """
    dev_metrics = Metrics('dev_loss', *loss_names, *monitor_names, data_type="avg")
    dev_metrics.reset()
    with torch.no_grad():
        model.eval()
        for j, dev_batch in enumerate(dev_it):
            R = model.gumbel_forward(dev_batch, en_lm=extra_input["en_lm"], all_img=extra_input["img"]['multi30k'][1],
                                     ranker=extra_input["ranker"])
            losses = [R[key] for key in loss_names]
            dev_metrics.accumulate(len(dev_batch), *[loss.item() for loss in losses],
                                   *[R[k].item() for k in monitor_names])
    return dev_metrics


def train_model(args, model, iterators, extra_input):
    (train_it, dev_it) = iterators

    if not args.debug:
        decoding_path = Path(args.decoding_path + args.id_str)
        decoding_path.mkdir(parents=True, exist_ok=True)
        from tensorboardX import SummaryWriter
        writer = SummaryWriter( args.event_path + args.id_str)

    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == 'Adam':
        opt = torch.optim.Adam(params, betas=(0.9, 0.98), eps=1e-9, lr=args.lr)
    else:
        raise NotImplementedError

    monitor_names = []
    loss_names = ['ce_loss']
    loss_cos = {"ce_loss": args.ce_co}

    if args.use_ranker:
        monitor_names.extend(["img_pred_loss_{}".format(args.img_pred_loss)])
    if args.use_en_lm:
        monitor_names.append('en_nll_lm')

    train_metrics = Metrics('train_loss', *loss_names, *monitor_names, data_type="avg")
    best = Best(max, 'de_bleu', 'en_bleu', 'iters', model=model, opt=opt, path=args.model_path + args.id_str, gpu=args.gpu, debug=args.debug)

    for iters, train_batch in enumerate(train_it):
        if iters >= args.max_training_steps:
            args.logger.info('stopping training after {} training steps'.format(args.max_training_steps))
            break

        if not args.debug and iters in args.save_at:
            args.logger.info('save (back-up) checkpoints at iters={}'.format(iters))
            with torch.cuda.device(args.gpu):
                torch.save(model.state_dict(), '{}_iter={}.pt'.format( args.model_path + args.id_str, iters))
                torch.save([iters, opt.state_dict()], '{}_iter={}.pt.states'.format( args.model_path + args.id_str, iters))

        if iters % args.eval_every == 0:
            dev_metrics = valid_model(model, dev_it, loss_names, monitor_names, extra_input)
            eval_metric, bleu_en, bleu_de = eval_model(args, model, dev_it, monitor_names, iters, extra_input)
            if not args.debug:
                write_tb(writer, loss_names, [dev_metrics.__getattr__(name) for name in loss_names],
                         iters, prefix="dev/")
                write_tb(writer, monitor_names, [dev_metrics.__getattr__(name) for name in monitor_names],
                         iters, prefix="dev/")

                write_tb(writer, ['bleu', *("p_1 p_2 p_3 p_4".split()), 'bp', 'len_ref', 'len_hyp'], bleu_en, iters,
                         prefix="bleu_en/")
                write_tb(writer, ['bleu', *("p_1 p_2 p_3 p_4".split()), 'bp', 'len_ref', 'len_hyp'], bleu_de, iters,
                         prefix="bleu_de/")
                write_tb(writer, ["bleu_en", "bleu_de"], [bleu_en[0], bleu_de[0]], iters, prefix="eval/")
                write_tb(writer, monitor_names, [eval_metric.__getattr__(name) for name in monitor_names],
                         iters, prefix="eval/")

            args.logger.info('model:' + args.prefix + args.hp_str)
            best.accumulate(bleu_de[0], bleu_en[0], iters)
            args.logger.info(best)
            if args.early_stop and (iters - best.iters) // args.eval_every > args.patience:
                args.logger.info("Early stopping.")
                break

        model.train()

        def get_lr_anneal(iters):
            lr_end = args.lr_min
            return max( 0, (args.lr - lr_end) * (args.linear_anneal_steps - iters) / args.linear_anneal_steps ) + lr_end

        def get_h_co_anneal(iters):
            h_co_end = args.h_co_min
            return max( 0, (args.h_co - h_co_end) * (args.h_co_anneal_steps - iters) / args.h_co_anneal_steps ) + h_co_end

        if args.lr_anneal == "linear":
            opt.param_groups[0]['lr'] = get_lr_anneal(iters)

        if args.h_co_anneal == "linear":
            loss_cos['neg_Hs'] = get_h_co_anneal(iters)

        opt.zero_grad()

        batch_size = len(train_batch)
        R = model.gumbel_forward(train_batch, en_lm=extra_input["en_lm"], all_img=extra_input["img"]['multi30k'][0], ranker=extra_input["ranker"])
        losses = [R[key] for key in loss_names]

        total_loss = 0
        for loss_name, loss in zip(loss_names, losses):
            total_loss += loss * loss_cos[loss_name]

        train_metrics.accumulate(batch_size, *[loss.item() for loss in losses], *[R[k].item() for k in monitor_names])

        total_loss.backward()
        if args.plot_grad:
            plot_grad(writer, model, iters)

        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(params, args.grad_clip)
        opt.step()

        if iters % args.eval_every == 0:
            args.logger.info("update {} : {}".format(iters, str(train_metrics)))

        if iters % args.eval_every == 0 and not args.debug:
            write_tb(writer, loss_names, [train_metrics.__getattr__(name) for name in loss_names], \
                     iters, prefix="train/")
            write_tb(writer, monitor_names, [train_metrics.__getattr__(name) for name in monitor_names], \
                     iters, prefix="train/")
            write_tb(writer, ['lr'], [opt.param_groups[0]['lr']], iters, prefix="train/")
            train_metrics.reset()

