"""
Test learning speed
"""
import sys
import torch
import os

import torch.nn.functional as F
from run_utils import get_model, get_data, get_ckpt_paths
from agent import ImageCaptioning, RNNLM, ImageGrounding
from imitate_utils import get_fr_en_imitate_stats, _get_imitate_loss
from hyperparams import Params
import random


# Parse from cmd to get JSON
parsed_args, _ = Params.parse_cmd(sys.argv)

must_includes = ['config', 'exp_dir', 'data_dir', 'fr_en_ckpt']
for must in must_includes:
    if must not in parsed_args:
        raise ValueError('You must provide --{}'.format(must))
print('Find json_config')
args = Params(parsed_args['config'])


# Update some of them with command line
args.update(parsed_args)

args.exp_dir = os.path.abspath(args.exp_dir)
main_path = args.exp_dir

# Data
train_it, dev_it = get_data(args)
EN = args.EN


if args.gpu > -1 and torch.cuda.device_count() > 0:
    map_location = lambda storage, loc: storage.cuda()
else:
    map_location = lambda storage, loc: storage


def get_extra_input(args):
    extra_input = {"en_lm": None, "img": {"multi30k": [None, None]}, "ranker": None, "s2p_it": None}
    lm_param, lm_model = get_ckpt_paths(args.exp_dir, args.lm_ckpt)
    print("Loading LM from: " + lm_param)
    args_ = Params(lm_param)
    LM_CLS = ImageCaptioning if args.en_lm_dataset in ["coco"] else RNNLM
    en_lm = LM_CLS(args_, len(args.EN.vocab.itos))
    en_lm.load_state_dict(torch.load(lm_model, map_location))
    en_lm.eval()
    if torch.cuda.device_count() > 0:
        en_lm.cuda(args.gpu)
    extra_input["en_lm"] = en_lm

    ranker_param, ranker_model = get_ckpt_paths(args.exp_dir, args.ranker_ckpt)
    print("Loading ranker from: " + ranker_param)
    args_ = Params(ranker_param)
    if args.img_pred_loss == "nll":
        ranker = ImageCaptioning(args_, len(args.EN.vocab.itos))
    else:
        ranker = ImageGrounding(args_, len(args.EN.vocab.itos))
    ranker.load_state_dict(torch.load(ranker_model, map_location) )
    ranker.eval()
    if torch.cuda.device_count() > 0:
        ranker.cuda(args.gpu)
    extra_input["ranker"] = ranker

    img = {}
    flickr30k_dir = os.path.join(args.data_dir, 'flickr30k')
    train_feats = torch.load(os.path.join(flickr30k_dir, 'train_feat.pth'))
    val_feats = torch.load(os.path.join(flickr30k_dir, 'val_feat.pth'))
    img["multi30k"] = [torch.tensor(train_feats), torch.tensor(val_feats)]
    print("Loading Flickr30k image features: train {} valid {}".format(
        img['multi30k'][0].shape, img['multi30k'][1].shape))
    extra_input["img"] = img
    return extra_input


extra_input = get_extra_input(args)


def main_loop(args, err_type):
    # Reload model
    model = get_model(args)
    model.fr_en.load_state_dict(torch.load(args.fr_en_ckpt, map_location))
    model.train(True)
    if args.gpu > -1 and torch.cuda.device_count() > 0:
        model = model.cuda(args.gpu)

    # Opt
    fr_en_opt = torch.optim.Adam(model.fr_en.parameters(), lr=args.fr_en_lr)
    monitor_names = []
    monitor_names.extend(["img_pred_loss_{}".format(args.img_pred_loss)])
    monitor_names.extend(["r1_acc"])
    monitor_names.append('en_nll_lm')

    statss = []
    eval_freq = min(max(int(args.fr_en_k2 / 30), 5), 50)
    iters = 0
    DAVID = 2464
    for j, batch in enumerate(train_it):
        if iters >= args.fr_en_k2:
            print('fr en stop learning after {} training steps'.format(args.fr_en_k2))
            break

        if iters % eval_freq == 0:
            print('Record stats at {}'.format(iters))
            model.eval()
            stats = get_fr_en_imitate_stats(args, model, dev_it, monitor_names, extra_input)
            statss.append((iters, stats))

        # Train
        model.train()
        src, src_len = batch.__dict__['fr']
        trg, trg_len = batch.__dict__['en']
        if err_type == 'second':
            # Insert at second position
            new_trg = torch.zeros(trg.shape[0], trg.shape[1] + 1).long().to(device=trg.device)
            for idx, sent in enumerate(trg):
                sent = sent.tolist()
                sent.insert(2, DAVID)
                new_trg[idx] = torch.LongTensor(sent)
        elif err_type == 'random':
            # Insert at random position
            new_trg = torch.zeros(trg.shape[0], trg.shape[1] + 1).long().to(device=trg.device)
            for idx, (sent, length) in enumerate(zip(trg, trg_len)):
                sent = sent.tolist()
                sent.insert(random.randint(1, length - 2), DAVID)
                new_trg[idx] = torch.LongTensor(sent)
        else:
            raise ValueError('Invalid err_type')
        trg_len += 1

        logits, _ = model.fr_en(src[:, 1:], src_len - 1, new_trg[:, :-1])
        nll = F.cross_entropy(logits, new_trg[:, 1:].contiguous().view(-1), reduction='mean',
                              ignore_index=0)

        fr_en_opt.zero_grad()
        nll.backward()
        fr_en_opt.step()
        iters += 1
    from tensorboardX import SummaryWriter
    from shutil import rmtree
    name = "{}_lr{}".format(err_type, args.fr_en_lr)
    print('save result to <{}>'.format(name))
    if os.path.exists(name):
        rmtree(name)
    tb_writer = SummaryWriter(name)
    for step, stats in statss:
        for key, val in stats.items():
            tb_writer.add_scalar(key, val, step)
    tb_writer.flush()


# Main loop
# Training
for err_type in ['random', 'second']:
    main_loop(args, err_type)
