import random
import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from data import NormalField, NormalTranslationDataset, TripleTranslationDataset
from utils import token_analysis, get_counts, write_tb, plot_grad, cuda
from metrics import Metrics, Best

from pathlib import Path

def valid_model(args, model, dev_it, dev_metrics, iters, loss_names, monitor_names, extra_input):
    with torch.no_grad():
        model.eval()

        for j, dev_batch in enumerate(dev_it):
            img_input = None if args.no_img else cuda(extra_input["img"]['multi30k'][1].index_select(dim=0, index=dev_batch.idx.cpu()))
            en, en_len = dev_batch.en
            decoded = model(en, img_input)
            R = {}
            R["nll"] = F.cross_entropy( decoded, en[:,1:].contiguous().view(-1), ignore_index=0 )
            #R["nll_cur"] = F.cross_entropy( decoded, en[:,:-1].contiguous().view(-1), ignore_index=0 )

            if not (img_input is None):
                idx = cuda(torch.randperm(img_input.size(0)))
                img_input = img_input.index_select(0, idx)
                decoded = model(en, img_input)
            R["nll_rnd"] = F.cross_entropy( decoded, en[:,1:].contiguous().view(-1), ignore_index=0 )

            dev_metrics.accumulate(len(dev_batch), *[R[name].item() for name in loss_names + monitor_names])

        args.logger.info(dev_metrics)

def train_model(args, model, iterators, extra_input):
    (train_its, dev_its) = iterators

    if not args.debug:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter( args.event_path + args.id_str)

    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == 'Adam':
        opt = torch.optim.Adam(params, betas=(0.9, 0.98), eps=1e-9, lr=args.lr)
    else:
        raise NotImplementedError

    loss_names, loss_cos = ["nll"], {"nll":1.0}
    monitor_names = ["nll_rnd"]
    """
    if args.rep_pen_co > 0.0:
        loss_names.append("nll_cur")
        loss_cos["nll_cur"] = -1 * args.rep_pen_co
    else:
        monitor_names.append("nll_cur")
    """

    train_metrics = Metrics('train_loss', *loss_names, data_type = "avg")
    dev_metrics = Metrics('dev_loss', *loss_names, *monitor_names, data_type = "avg")
    best = Best(min, 'loss', 'iters', model=model, opt=opt, path=args.model_path + args.id_str, \
                gpu=args.gpu, debug=args.debug)

    iters = 0
    should_stop = False
    for epoch in range(999999999):
        if should_stop:
            break

        for dataset in args.dataset.split("_"):
            if should_stop:
                break

            train_it = train_its[dataset]
            for _, train_batch in enumerate(train_it):
                if iters >= args.max_training_steps:
                    args.logger.info('stopping training after {} training steps'.format(args.max_training_steps))
                    should_stop = True
                    break

                if iters % args.eval_every == 0:
                    dev_metrics.reset()
                    valid_model(args, model, dev_its['multi30k'], dev_metrics, iters, loss_names, monitor_names, extra_input)
                    if not args.debug:
                        write_tb(writer, loss_names, [dev_metrics.__getattr__(name) for name in loss_names], \
                                 iters, prefix="dev/")
                        write_tb(writer, monitor_names, [dev_metrics.__getattr__(name) for name in monitor_names], \
                                 iters, prefix="dev/")
                    best.accumulate(dev_metrics.nll, iters)

                    args.logger.info('model:' + args.prefix + args.hp_str)
                    args.logger.info('epoch {} dataset {} iters {}'.format(epoch, dataset, iters))
                    args.logger.info(best)

                    if args.early_stop and (iters - best.iters) // args.eval_every > args.patience:
                        args.logger.info("Early stopping.")
                        return

                model.train()

                def get_lr_anneal(iters):
                    lr_end = args.lr_min
                    return max( 0, (args.lr - lr_end) * (args.linear_anneal_steps - iters) / args.linear_anneal_steps ) + lr_end

                if args.lr_anneal == "linear":
                    opt.param_groups[0]['lr'] = get_lr_anneal(iters)

                opt.zero_grad()

                batch_size = len(train_batch)
                img_input = None if args.no_img else cuda(extra_input["img"][dataset][0].index_select(dim=0, index=train_batch.idx.cpu()))
                if dataset == "coco":
                    en, en_len = train_batch.__dict__["_"+str(random.randint(1,5))]
                elif dataset == "multi30k":
                    en, en_len = train_batch.en

                decoded = model(en, img_input)
                R = {}
                R["nll"] = F.cross_entropy( decoded, en[:,1:].contiguous().view(-1), ignore_index=0 )
                #R["nll_cur"] = F.cross_entropy( decoded, en[:,:-1].contiguous().view(-1), ignore_index=0 )

                total_loss = 0
                for loss_name in loss_names:
                    total_loss += R[loss_name] * loss_cos[loss_name]

                train_metrics.accumulate(batch_size, *[R[name].item() for name in loss_names])

                total_loss.backward()
                if args.plot_grad:
                    plot_grad(writer, model, iters)

                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(params, args.grad_clip)

                opt.step()
                iters += 1

                if iters % args.eval_every == 0:
                    args.logger.info("update {} : {}".format(iters, str(train_metrics)))

                if iters % args.eval_every == 0 and not args.debug:
                    write_tb(writer, loss_names, [train_metrics.__getattr__(name) for name in loss_names], \
                             iters, prefix="train/")
                    write_tb(writer, ['lr'], [opt.param_groups[0]['lr']], iters, prefix="train/")
                    train_metrics.reset()

