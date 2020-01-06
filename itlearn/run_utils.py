from agent import RNNAttn, RNNLM, ImageCaptioning, ImageGrounding
from agents import Agents
from data import build_dataset
import os


def get_model(args):
    if args.setup == "joint":
        model = Agents(args)
    elif args.setup == 'itlearn':
        model = Agents(args)
    elif args.setup == "single":
        model = RNNAttn(args, args.voc_sz_src, args.voc_sz_trg)
    elif args.setup == "lm":
        model = RNNLM(args, len(args.EN.vocab.itos))
    elif args.setup == "ranker":
        if args.img_pred_loss == "nll":
            model = ImageCaptioning(args, len(args.EN.vocab.itos))
        else:
            model = ImageGrounding(args, len(args.EN.vocab.itos))
    else:
        raise ValueError
    return model


def get_data(args):
    if args.setup == "ranker":
        train_it, dev_it = {}, {}
        for dataset in ["coco", "multi30k"]:
            train_it_, dev_it_ = build_dataset(args, dataset)
            train_it[dataset] = train_it_
            dev_it[dataset] = dev_it_
    else:
        train_it, dev_it = build_dataset(args, args.dataset)
    return train_it, dev_it


def get_ckpt_paths(exp_dir, ckpt, cpt_iter="best"):
    load_param_from = os.path.join(exp_dir, 'param', ckpt)
    load_model_from = os.path.join(exp_dir, 'model', ckpt)
    load_model_from = "{}_best.pt".format(load_model_from) if cpt_iter == "best" \
        else "{}_iter={}.pt".format(load_model_from, cpt_iter)
    return load_param_from, load_model_from
