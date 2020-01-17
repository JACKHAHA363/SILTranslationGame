import random
import math
import os

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from data import NormalField, NormalTranslationDataset, TripleTranslationDataset, TextVocab
from utils import token_analysis, get_counts, write_tb, plot_grad, cuda, normf
from metrics import Metrics, Best

import torchvision
from img_utils import preprocess_1c, ImageFolderWithPaths, preprocess_rc

def retrieval(idx, query, dim):
    # idx : (K, nb_qury) or (nb_query, K)
    # query : (query)
    # dim = 0 or 1
    if dim == 0:
        (K, batch_size) = idx.shape
        query = query[None,:].repeat(K,1)
        match = (idx == query).float().sum(dim=dim) # (1000)
        match = torch.clamp(match, 0, 1)
        match = match.mean()
    else:
        (batch_size, K) = idx.shape
        query = query[:,None].repeat(1,K)
        match = (idx == query).float().sum(dim=dim) # (1000)
        match = torch.clamp(match, 0, 1)
        match = match.mean()
    return match.item()


def get_retrieve_result(args, model, valid_img_feats, valid_caps, valid_lens):
    with torch.no_grad():
        model.eval()
        query = cuda(torch.arange(valid_img_feats.shape[0]))
        img_features = model.batch_enc_img(valid_img_feats)
        cap_features = []
        for valid_cap, valid_len in zip(valid_caps, valid_lens):
            cap_features.append(model.batch_cap_rep(valid_cap[:, 1:], valid_len - 1))
        scores = [torch.mm(img_features, cap.t()) for cap in cap_features] # [ (img 1000, cap_i 1000) x 5 ]
        img_idxs = [torch.sort( sc, dim=0, descending=True )[1] for sc in scores] # [ (img 1000, cap_i 1000) x 5 ]
        img_r1  = np.mean([retrieval(img_idx[:1, :], query, 0) for img_idx in img_idxs])
        img_r5  = np.mean([retrieval(img_idx[:5, :], query, 0) for img_idx in img_idxs])
        img_r10 = np.mean([retrieval(img_idx[:10, :], query, 0) for img_idx in img_idxs])

        scores = torch.cat(scores, dim=1 ) # (img 1000, cap 5000)
        cap_idx = torch.sort(scores, dim=1, descending=True )[1] # (img 1000, cap 5000)
        cap_idx = torch.remainder(cap_idx, 1000 ) # (img 1000, cap 5000)
        cap_r1, cap_r5, cap_r10 = [retrieval(idx, query, 1) for idx in \
                                   [ cap_idx[:,:1], cap_idx[:,:5], cap_idx[:,:10] ] ]
        return cap_r1, cap_r5, cap_r10, img_r1, img_r5, img_r10


def valid_model(args, model, valid_img_feats, valid_caps, valid_lens):
    model.eval()
    batch_size = 32
    start = 0
    val_metrics = Metrics('val_loss', 'loss', data_type="avg")
    with torch.no_grad():
        while start <= valid_img_feats.shape[0]:
            cap_id = random.randint(0, 4)
            end = start + batch_size
            batch_img_feat = cuda(valid_img_feats[start: end])
            batch_ens = cuda(valid_caps[cap_id][start: end])
            batch_lens = cuda(valid_lens[cap_id][start: end])
            R = model(batch_ens[:, 1:], batch_lens - 1, batch_img_feat)
            if args.img_pred_loss == "vse":
                R['loss'] = R['loss'].sum()
            elif args.img_pred_loss == "mse":
                R['loss'] = R['loss'].mean()
            else:
                raise ValueError
            val_metrics.accumulate(batch_size, R['loss'])
            start = end
        return val_metrics


def train_model(args, model):

    resnet = torchvision.models.resnet152(pretrained=True)
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    resnet = nn.DataParallel(resnet).cuda()
    resnet.eval()

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

    train_metrics = Metrics('train_loss', *loss_names, data_type="avg")
    best = Best(max, 'r1', 'iters', model=model, opt=opt, path=args.model_path + args.id_str, \
                gpu=args.gpu, debug=args.debug)

    # Train dataset
    args.logger.info("Loading train imgs...")
    train_dataset = ImageFolderWithPaths(os.path.join(args.data_dir, 'flickr30k'), preprocess_rc)
    train_imgs = open(os.path.join(args.data_dir, 'flickr30k/train.txt'), 'r').readlines()
    train_imgs = [x.strip() for x in train_imgs if x.strip() != ""]
    train_dataset.samples = [x for x in train_dataset.samples if x[0].split("/")[-1] in train_imgs]
    train_dataset.imgs = [x for x in train_dataset.imgs if x[0].split("/")[-1] in train_imgs]
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8,
                                               pin_memory=False)
    args.logger.info("Train loader built!")

    en_vocab = TextVocab(counter=torch.load(os.path.join(args.data_dir, 'bpe/vocab.en.pth')))
    word2idx = en_vocab.stoi
    train_en = [open(os.path.join(args.data_dir, 'flickr30k/caps', 'train.{}.bpe'.format(idx+1))).readlines() for idx in range(5)]
    train_en = [[["<bos>"] + sentence.strip().split() + ["<eos>"] for sentence in doc if sentence.strip() != "" ] for doc in train_en]
    train_en = [[[word2idx[word] for word in sentence] for sentence in doc] for doc in train_en]

    args.logger.info("Train corpus built!")

    # Valid dataset
    valid_img_feats = torch.tensor(torch.load(os.path.join(args.data_dir, 'flickr30k/val_feat.pth')))
    valid_ens = []
    valid_en_lens = []
    for idx in range(5):
        valid_en = []
        with open(os.path.join(args.data_dir, 'flickr30k/caps', 'val.{}.bpe'.format(idx+1))) as f:
            for line in f:
                line = line.strip()
                if line == "":
                    continue

                words = ["<bos>"] + line.split() + ["<eos>"]
                words = [word2idx[word] for word in words]
                valid_en.append(words)

        # Pad
        valid_en_len = [len(sent) for sent in valid_en]
        valid_en = [np.lib.pad(xx, (0, max(valid_en_len) - len(xx)), 'constant', constant_values=(0, 0)) for xx in valid_en]
        valid_ens.append(torch.tensor(valid_en).long())
        valid_en_lens.append(torch.tensor(valid_en_len).long())
    args.logger.info("Valid corpus built!")

    iters = -1
    should_stop = False
    for epoch in range(999999999):
        if should_stop:
            break

        for idx, (train_img, lab, path) in enumerate(train_loader):
            iters += 1
            if iters > args.max_training_steps:
                should_stop = True
                break

            if iters % args.eval_every == 0:
                res = get_retrieve_result(args, model, valid_caps=valid_ens, valid_lens=valid_en_lens,
                                          valid_img_feats=valid_img_feats)
                val_metrics = valid_model(args, model, valid_img_feats=valid_img_feats,
                                          valid_caps=valid_ens, valid_lens=valid_en_lens)
                args.logger.info("[VALID] update {} : {}".format(iters, str(val_metrics)))
                if not args.debug:
                    write_tb(writer, monitor_names, res, iters, prefix="dev/")
                    write_tb(writer, loss_names, [val_metrics.__getattr__(name) for name in loss_names],
                             iters, prefix="dev/")

                best.accumulate((res[0]+res[3])/2, iters)
                args.logger.info('model:' + args.prefix + args.hp_str)
                args.logger.info('epoch {} iters {}'.format(epoch, iters))
                args.logger.info(best)

                if args.early_stop and (iters - best.iters) // args.eval_every > args.patience:
                    args.logger.info("Early stopping.")
                    return

            model.train()

            def get_lr_anneal(iters):
                lr_end = args.lr_min
                return max( 0, (args.lr - lr_end) * (args.linear_anneal_steps - iters) /
                           args.linear_anneal_steps ) + lr_end

            if args.lr_anneal == "linear":
                opt.param_groups[0]['lr'] = get_lr_anneal(iters)

            opt.zero_grad()
            batch_size = len(path)
            path = [p.split("/")[-1] for p in path]
            sentence_idx = [train_imgs.index(p) for p in path]
            en = [train_en[random.randint(0, 4)][sentence_i] for sentence_i in sentence_idx]
            en_len = [len(x) for x in en]

            en = [ np.lib.pad( xx, (0, max(en_len) - len(xx)), 'constant', constant_values=(0,0) ) for xx in en ]
            en = cuda( torch.LongTensor( np.array(en).tolist() ) )
            en_len = cuda( torch.LongTensor( en_len ) )

            with torch.no_grad():
                train_img = resnet(train_img).view(batch_size, -1)
            R = model(en[:,1:], en_len-1, train_img)
            if args.img_pred_loss == "vse":
                R['loss'] = R['loss'].sum()
            elif args.img_pred_loss == "mse":
                R['loss'] = R['loss'].mean()
            else:
                raise Exception()

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

            if iters % args.eval_every == 0:
                args.logger.info("update {} : {}".format(iters, str(train_metrics)))

            if iters % args.eval_every == 0 and not args.debug:
                write_tb(writer, loss_names, [train_metrics.__getattr__(name) for name in loss_names], \
                         iters, prefix="train/")
                write_tb(writer, ['lr'], [opt.param_groups[0]['lr']], iters, prefix="train/")
                train_metrics.reset()

