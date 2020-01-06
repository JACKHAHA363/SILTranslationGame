import logging
import math
import ipdb
import random
import numpy as np
import _pickle as pickle
import revtok
import os
from itertools import groupby
import getpass
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from torchtext import data, datasets
from collections import OrderedDict
import fractions

try:
    fractions.Fraction(0, 1000, _normalize=False)
    from fractions import Fraction
except TypeError:
    from nltk.compat import Fraction

def write_tb(writer, keys, values, idx, prefix=""):
    for k, v in zip(keys, values):
        writer.add_scalar(prefix+k, v, idx)

def plot_grad(writer, model, idx):
    keys, values = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            keys.append(name)
            values.append(param.grad.norm().item())
    if keys:
        write_tb(writer, keys, values, idx, prefix="grad/")

def get_counts(en_msg, seq_lens, vocab_size):
    # en_msg : (batch_size, seq_len)
    # seq_lens : batch_size
    (batch_size, seq_len) = en_msg.size()
    en_msg_onehot = cuda( torch.zeros(batch_size, seq_len, vocab_size) )
    en_msg_onehot.scatter_(2, en_msg[:,:,None], 1)

    mask = cuda( torch.full((batch_size, seq_len), 1).byte() )
    for idx in range(batch_size):
        mask[idx][:seq_lens[idx]] = 0

    en_msg_onehot.masked_fill_(mask[:,:,None], 0)

    C = en_msg_onehot.sum(dim=1).sum(dim=0) # (vocab_size)
    return C

def token_analysis(C):
    C = C.int()
    num_tokens = C.sum().int().item()
    num_unique_tokens = (C != 0).sum().item()
    sorted_C, indices = torch.sort(C, descending=True)
    top, middle, bottom = sorted_C[0].int().item(), sorted_C[num_unique_tokens//2].int().item(), sorted_C[num_unique_tokens-1].int().item()

    return (num_tokens, num_unique_tokens, top, middle, bottom)

def take_last(output, x_len, num_dir_enc, D_hid_enc):
    batch_size = x_len.size()[0]
    x_seq_len = x_len.max().item()

    if num_dir_enc == 1:
        index = (x_len - 1)[:,None,None].repeat(1, 1, output.size(2))
        # index : (batch_size, 1 , num_dir * D_hid)
        out = output.gather(dim=1, index=index).view(batch_size, -1)

    elif num_dir_enc == 2:
        index = (x_len - 1)[:,None,None].repeat(1, 1, output.size(2) / 2)
        # index : (batch_size, 1 , D_hid)
        output = output.view(batch_size, x_seq_len, 2, D_hid_enc)
        fwd = output[:,:,0,:].gather(dim=1, index=index).view(batch_size, -1) # (batch_size, D_hid)
        bwd = output[:,0,1,:] # (batch_size, D_hid)
        out = torch.cat([fwd, bwd], dim=1)

    return out # (batch_size, num_dir * D_hid)

def lm_forward(model, input, hidden):
    # input : (seq_len, batch_size)
    # hidden : (num_layers, batch_size, D_hid)
    emb = model.encoder(input)
    output, hidden = model.rnn(emb, hidden)
    decoded = model.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
    return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

def lm_reverse(idx2word, tensor):
    return " ".join( [idx2word[idx] for idx in tensor.cpu().numpy()] )

def get_nll_lm_bpe(model, en_msg, en_msg_len, nhid=400):
    # msg : (batch_size)
    with torch.no_grad():
        batch_size, en_msg_max_len = en_msg.size()
        data = en_msg.t()
        mask = xlen_to_inv_mask(en_msg_len-1, en_msg.size(1)-1)

        input, target = data[:-1,:].contiguous(), data[1:,:].contiguous()
        # (en_msg_max_len - 1, batch_size)
        hidden = ( cuda( torch.FloatTensor( 2, batch_size, nhid ).zero_() ), \
                   cuda( torch.FloatTensor( 2, batch_size, nhid ).zero_() ) )
        output, _ = lm_forward(model, input, hidden )
        # (en_msg_max_len-1, batch_size, voc_size)
        logits = output.contiguous().view( -1, output.size()[-1] )
        # (en_msg_max_len-1 * batch_size, voc_size)
        nll = F.cross_entropy( logits, target.view(-1), ignore_index=0, reduction='none') # (en_msg_len-1 * batch_size)
        nll = nll.view(-1, batch_size).t().contiguous() # (batch_size, en_msg_len-1)
        nll.masked_fill_(mask, 0)
        nll = nll.sum(dim=-1) / (en_msg_len-1).float() # (batch_size)
        return nll # (batch_size)

def get_nll_lm_char(model, en_msg, word2idx, idx2word, nhid=800):
    # msg : (batch_size)
    with torch.no_grad():
        batch_size = len(en_msg)
        num_chars = cuda( torch.FloatTensor( [len(msg) for msg in en_msg] ) )
        #assert (min([len(msg) for msg in en_msg]) > 1)
        #msgs = [msg.split() + ["<eos>"] for msg in msgs]
        msgs_idx = [ [ word2idx[tok] if tok in word2idx else word2idx["<unk>"] for tok in msg ] for msg in en_msg]
        msgs_len = [len(msg) for msg in msgs_idx]
        msgs_idx = [ np.lib.pad( msg, (0, max(msgs_len) - ln), 'constant', constant_values=(0, 0) ) for (msg, ln) in zip(msgs_idx, msgs_len) ]
        data = cuda( torch.LongTensor(msgs_idx) ).t() # (seq_len, batch_size)
        seq_len = data.size(0)

        input, target = data[:-1,:].contiguous(), data[1:,:].contiguous()
        hidden = ( cuda( torch.FloatTensor( 2, batch_size, nhid ).zero_() ), \
                   cuda( torch.FloatTensor( 2, batch_size, nhid ).zero_() ) )
        output, _ = lm_forward(model, input, hidden )
        # output : (seq_len-1, batch_size, voc_size)
        logits = output.contiguous().view( -1, output.size()[-1] )
        nll = F.cross_entropy( logits, target.view(-1), ignore_index=0, reduce=False) # (batch_size, en_msg_len)
        nll = nll.view(-1, data.size(1)).t().contiguous() # (seq_len-1, batch_size)
        nll = nll.sum(dim=-1) / num_chars # (batch_size)
        return nll

def xlen_to_mask(x_len):
    # x_len : (batch_size)
    batch_size, seq_len = x_len.size()[0], x_len.max()

    mask = torch.full((batch_size, seq_len), 0).byte()
    for idx in range(batch_size):
        mask[idx][:x_len[idx]] = 1
    return cuda(mask)

def xlen_to_inv_mask(x_len, seq_len=None):
    # x_len : (batch_size)
    batch_size = x_len.size()[0]
    if seq_len is None:
        seq_len = x_len.max()

    mask = torch.full((batch_size, seq_len), 1).byte()
    for idx in range(batch_size):
        mask[idx][:x_len[idx]] = 0
    return cuda(mask)

def sample_gumbel(shape, eps=1e-20):
    U = cuda( torch.FloatTensor(shape).uniform_(0, 1) )
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax(logits, temp):
    y = ( logits + sample_gumbel(logits.size()) ) / temp
    return F.softmax(y, dim=1)

def gumbel_softmax_hard(logits, temp, st):
    y = gumbel_softmax(logits, temp) # (batch_size, num_cat)
    y_max, y_max_idx = torch.max(y, 1, keepdim=True) # NOTE non-differentiable
    if st:
        y_hard = cuda( torch.FloatTensor(y.size()) ).zero_().scatter_(1, y_max_idx.data, 1)
        y = y_hard - y.data + y
    return y

def cuda(x):
    if torch.cuda.is_available():
        return x.cuda()
    return x

def which_machine():
    if "mila" in os.uname()[1]:
        return "mila"
    else:
        raise Exception

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.deterministic and torch.backends.cudnn.enabled:
        torch.backends.cudnn.deterministic = True

def get_dist_idx(batch_size, num_dist):
    result = []
    for idx in range(num_dist):
        idx = torch.remainder( torch.arange(batch_size) + idx, batch_size ).long()
        result.append(idx)
    return cuda(torch.stack(result, dim=1)) # (batch_size, num_dist)

def normf(t, p=2, d=1):
    return t / t.norm(p, d, keepdim=True).expand_as(t)

def sum_reward(reward, lens):
    # reward: (batch_size, seq_len)
    # lens : (batch_size)
    mask = xlen_to_inv_mask(lens, reward.size(1))
    reward.masked_fill_(mask.bool(), 0)
    reward = reward.sum(dim=-1) / lens.float() # (batch_size)
    return reward

def get_logger(args):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(message)s')

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if not args.debug:
        fh = logging.FileHandler( args.log_path + args.id_str )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger
