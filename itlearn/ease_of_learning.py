"""
Evaluate the ease of teaching accross the training trajectory
"""
import argparse
from models.agent import RNNAttn
import torch
import os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('decoding_dir')
    parser.add_argument('-gpu', action='store_true')
    return parser.parse_args()


def get_fr_en_model(voc_sz_src, voc_sz_trg, ckpt_path=None):
    args = argparse.Namespace
    args.D_emb = 256
    args.D_hid = 256
    args.n_layers = 1
    args.n_dir = 1
    args.input_feeding = True
    args.tie_emb = False
    model = RNNAttn(args, voc_sz_src=voc_sz_src,
                    voc_sz_trg=voc_sz_trg)
    if ckpt_path is None:
        model.load_state_dict(torch.load(ckpt_path))
    return model


def main():
    args = get_args()
    en_files = os.listdir(args.decoding_dir)

    model = RNNAttn(args, args.voc_sz_src, args.voc_sz_trg)
