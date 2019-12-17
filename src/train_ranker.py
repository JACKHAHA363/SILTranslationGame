import random
import ipdb
import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from data import NormalField, NormalTranslationDataset, TripleTranslationDataset, data_path
from utils import token_analysis, get_counts, write_tb, plot_grad, cuda, normf
from metrics import Metrics, Best

from pathlib import Path

def retrieval(idx, query, dim):
    # idx : (K, 5000) or (5000, K)
    # query : (5000)
    # dim = 0 or 1
    if dim == 0:
        (K, batch_size) = idx.shape
        query = query[None,:].repeat(K,1)
        match = (idx == query).float().sum(dim=dim) # (5000)
        match = match.mean()
    else:
        (batch_size, K) = idx.shape
        query = query[:,None].repeat(1,K)
        match = (idx == query).float().sum(dim=dim) # (5000)
        match = match.mean()
    return match.item()

def valid_model(args, model, dev_it, extra_input):
    with torch.no_grad():
        model.eval()
        img_features, cap_features = [], []
        for j, dev_batch in enumerate(dev_it):
            img = cuda( extra_input["img"]["multi30k"][1].index_select(dim=0, index=dev_batch.idx.cpu())) # (batch_size, D_img)
            en, en_len = dev_batch.en
            cap_rep = model.get_cap_rep(en[:,1:], en_len-1)
            cap_features.append( normf(cap_rep) )
            img = model.img_enc( img ) # (batch_size, D_hid)
            img_features.append( normf(img) )
        img_features = torch.cat( img_features, dim=0 ) # (5000, D_img)
        cap_features = torch.cat( cap_features, dim=0 ) # (5000, D_img)
        scores = torch.mm( img_features, cap_features.t() ) # (5000, 5000)
        _, cap_idx = torch.sort( scores, dim=1, descending=True ) # (5000, 5000)
        _, img_idx = torch.sort( scores, dim=0, descending=True ) # (5000, 5000)
        query = cuda( torch.arange(5000) )
        cap_r1, cap_r5, cap_r10 = [retrieval(idx, query, 1) for idx in \
                                   [ cap_idx[:,:1], cap_idx[:,:5], cap_idx[:,:10] ] ]
        img_r1, img_r5, img_r10 = [retrieval(idx, query, 0) for idx in \
                                   [ img_idx[:1,:], img_idx[:5,:], img_idx[:10,:] ] ]
        return (cap_r1, cap_r5, cap_r10, img_r1, img_r5, img_r10)

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

    loss_names, loss_cos = ["loss"], {"loss":1.0}
    monitor_names = "cap_r1 cap_r5 cap_r10 img_r1 img_r5 img_r10".split()

    train_metrics = Metrics('train_loss', *loss_names, data_type = "avg")
    best = Best(max, 'r1', 'iters', model=model, opt=opt, path=args.model_path + args.id_str, \
                gpu=args.gpu, debug=args.debug)

    iters = 0
    for epoch in range(999999999):
        for dataset in args.dataset.split("_"):
            train_it = train_its[dataset]
            for _, train_batch in enumerate(train_it):
                iters += 1

                if iters % args.eval_every == 0:
                    R = valid_model(args, model, dev_its['multi30k'], extra_input)
                    if not args.debug:
                        write_tb(writer, monitor_names, R, iters, prefix="dev/")
                    best.accumulate((R[0]+R[3])/2, iters)

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
                img = extra_input["img"][dataset][0].index_select(dim=0, index=train_batch.idx.cpu()) # (batch_size, D_img)
                en, en_len = train_batch.__dict__["_"+str(random.randint(1,5))]
                R = model(en[:,1:], en_len-1, cuda(img))
                R['loss'] = R['loss'].mean()

                total_loss = 0
                for loss_name in loss_names:
                    total_loss += R[loss_name] * loss_cos[loss_name]

                train_metrics.accumulate(batch_size, *[R[name].item() for name in loss_names])

                total_loss.backward()
                if args.plot_grad:
                    plot_grad(writer, model, iters)

                if args.grad_clip > 0:
                    total_norm = nn.utils.clip_grad_norm_(params, args.grad_clip)
                    if total_norm != total_norm or math.isnan(total_norm) or np.isnan(total_norm):
                        ipdb.set_trace()

                opt.step()

                if iters % args.eval_every == 0:
                    args.logger.info("update {} : {}".format(iters, str(train_metrics)))

                if iters % args.eval_every == 0 and not args.debug:
                    write_tb(writer, loss_names, [train_metrics.__getattr__(name) for name in loss_names], \
                             iters, prefix="train/")
                    write_tb(writer, ['lr'], [opt.param_groups[0]['lr']], iters, prefix="train/")
                    train_metrics.reset()

