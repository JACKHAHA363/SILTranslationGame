import pprint
import ipdb
import operator
import torch
import math
import sys
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

from agent import RNN, RNNAttn

from utils import cuda, gumbel_softmax, gumbel_softmax_hard, xlen_to_mask, xlen_to_inv_mask, get_counts, \
        sum_reward
from modules import ArgsModule

SMALL = 1e-10

class Agents(ArgsModule):
    def __init__(self, args):
        super(Agents, self).__init__(args)

        #if args.model == "RNN":
        #    rnncls = RNN
        #elif args.model == "RNNAttn":
        #    rnncls = RNNAttn
        rnncls = RNNAttn

        self.fr_en = rnncls(args, len(args.FR.vocab.itos), len(args.EN.vocab.itos))
        self.en_de = rnncls(args, len(args.EN.vocab.itos), len(args.DE.vocab.itos))
        # NOTE Let the second En-De agent predict <EOS> symbol wherever it sees fit.
        # i.e. set max_len_gen=50 for En-De agent.
        self.en_de.dec.msg_len_ratio = -1.0

        value_fn_input_dim = args.D_hid + args.D_emb
        self.value_fn = nn.Sequential(
            nn.Linear(value_fn_input_dim, args.D_hid),
            nn.ReLU(),
            nn.Linear(args.D_hid, 1)
        )

    def eval_fr_en_stats(self, en_msg, en_msg_len, batch, en_lm=None, all_img=None, ranker=None):
        results = {}
        rewards = {}
        batch_size = en_msg.shape[0]
        # NOTE add <BOS> to beginning
        en_msg_ = torch.cat([cuda(torch.full((batch_size, 1), self.init_token)).long(), en_msg], dim=1)
        if self.use_en_lm: # monitor EN LM NLL
            if "wiki" in self.en_lm_dataset:
                en_nll_lm = en_lm.get_nll(en_msg_) # (batch_size, en_msg_len)
                en_nll_lm = sum_reward(en_nll_lm, en_msg_len + 1) # (batch_size)
                if self.train_en_lm:
                    rewards['lm'] = -1 * en_nll_lm.detach()
                    #R = R + -1 * en_nll_lm.detach() * self.en_lm_nll_co # (batch_size)
                results.update({"en_nll_lm": en_nll_lm.mean()})

            elif self.en_lm_dataset in ["coco", "multi30k"]:
                en_nll_lm = en_lm.get_loss( en_msg_, None ) # (batch_size, en_msg_len)
                en_nll_lm = sum_reward(en_nll_lm, en_msg_len + 1) # (batch_size)
                if self.train_en_lm:
                    rewards['lm'] = -1 * en_nll_lm.detach()
                results.update({"en_nll_lm": en_nll_lm.mean()})
            else:
                raise Exception()

        if self.use_ranker: # NOTE Experiment 3 : Reward = NLL_DE + NLL_EN_LM + NLL_IMG_PRED
            img = cuda(all_img.index_select(dim=0, index=batch.idx.cpu()) ) # (batch_size, D_img)

            if self.img_pred_loss == "nll":
                img_pred_loss = ranker.get_loss(en_msg_, img) # (batch_size, en_msg_len)
                img_pred_loss = sum_reward(img_pred_loss, en_msg_len + 1) # (batch_size)
            else:
                with torch.no_grad():
                    img_pred_loss = ranker(en_msg, en_msg_len, img)["loss"]

            if self.train_ranker:
                rewards['img_pred'] = -1 * img_pred_loss.detach()
            results.update({"img_pred_loss_{}".format(self.img_pred_loss): img_pred_loss.mean()})
        return results, rewards

    def selfplay_batch(self, batch, en_lm=None, all_img=None, ranker=None):
        """ Return all stuff related to reinforce """
        results = {}
        rewards = {}
        (fr, fr_len) = batch.fr
        (en, en_len) = batch.en
        (de, de_len) = batch.de
        batch_size = len(batch)

        # <BOS> removed from source Fr sentences when training Fr->En agent, hence fr_len - 1
        fr_hid = self.fr_en.enc(fr[:, 1:], fr_len-1)
        # Need to predict everything except <BOS>, hence en_len-1
        send_results = self.fr_en.dec.send(fr_hid, fr_len-1, en_len-1, "reinforce", self.value_fn)
        en_msg, en_msg_len = [send_results[key] for key in ["msg", "new_seq_lens"]]
        results.update(send_results)

        de_input, de_target = de[:, :-1], de[:, 1:].contiguous().view(-1)
        de_logits, _ = self.en_de(en_msg, en_msg_len, de_input) # (batch_size * en_seq_len, vocab_size)
        de_nll = F.cross_entropy(de_logits, de_target, ignore_index=0, reduction='none')
        results['ce_loss'] = de_nll.mean()
        de_nll = de_nll.view(batch_size, -1).sum(dim=1) / (de_len - 1).float() # (batch_size)
        rewards['ce'] = -1 * de_nll.detach() # NOTE Experiment 1 : Reward = NLL_DE

        # NOTE add <BOS> to beginning
        fr_en_results, fr_en_rewards = self.eval_fr_en_stats(en_msg, en_msg_len, batch, en_lm=en_lm,
                                                             all_img=all_img, ranker=ranker)
        results.update(fr_en_results)
        rewards.update(fr_en_rewards)

        if not (self.fr_en.dec.neg_Hs is []):
            neg_Hs = self.fr_en.dec.neg_Hs # (batch_size, en_msg_len)
            neg_Hs = neg_Hs.mean() # (1,)
            results["neg_Hs"] = neg_Hs
        return results, rewards

    def forward(self, batch, en_lm=None, all_img=None, ranker=None):
        """ Reinforce. """
        results, rewards = self.selfplay_batch(batch, en_lm, all_img, ranker)
        en_msg_len = results['new_seq_lens']
        R = rewards['ce']
        if self.train_en_lm:
            R += rewards['lm'] * self.en_lm_nll_co
        if self.train_ranker:
            R += rewards['img_pred'] * self.img_pred_loss_co

        if not self.fix_fr2en:
            R_b = self.fr_en.dec.R_b # (batch_size, en_msg_len)
            en_mask = xlen_to_inv_mask(en_msg_len, R_b.size(1))
            b_loss = ((R[:, None] - R_b) ** 2) # (batch_size, en_msg_len)
            b_loss.masked_fill_(en_mask.bool(), 0) # (batch_size, en_msg_len)
            b_loss = b_loss.sum(dim=1) / (en_msg_len).float() # (batch_size)
            b_loss = b_loss.mean() # (1,)
            pg_loss = -1 * self.fr_en.dec.log_probs # (batch_size, en_msg_len)
            pg_loss = (R[:,None] - R_b).detach() * pg_loss # (batch_size, en_msg_len)
            pg_loss.masked_fill_(en_mask.bool(), 0) # (batch_size, en_msg_len)
            pg_loss = pg_loss.sum(dim=1) / (en_msg_len).float() # (batch_size)
            pg_loss = pg_loss.mean() # (1,)
            results.update({"pg_loss": pg_loss, "b_loss": b_loss})
        return results

    def decode(self, batch, en_method='argmax', de_method='argmax'):
        (fr, fr_len) = batch.fr
        (en, en_len) = batch.en
        (de, de_len) = batch.de
        batch_size = len(batch)

        fr_hid = self.fr_en.enc(fr[:,1:], fr_len-1)
        en_send_results = self.fr_en.dec.send(fr_hid, fr_len-1, en_len-1, en_method)
        en_msg, en_msg_len = [en_send_results[key] for key in ["msg", "new_seq_lens"]]
        # NOTE new_seq_lens : give ground truth EN REF length
        # en_msg : (batch_size, seq_len)
        # en_msg_len : (batch_size)

        #inv_mask = xlen_to_inv_mask(en_msg_len)
        #en_msg.masked_fill_(mask=inv_mask, value=0) # NOTE make sure pads are really <PAD>

        en_hid = self.en_de.enc(en_msg, en_msg_len)
        de_send_results = self.en_de.dec.send(en_hid, en_msg_len, de_len-1, de_method)
        de_msg, de_msg_len = [de_send_results[key] for key in ["msg", "new_seq_lens"]]
        # NOTE seq_lens : give 50, terminate whenever En->De model outputs <EOS>

        return en_msg, de_msg, en_msg_len, de_msg_len

    def multi_decode(self, batch):
        (fr, fr_len) = batch.fr
        (en, en_len) = batch.en
        (de, de_len) = batch.de
        batch_size = len(batch)

        fr_hid = self.fr_en.enc(fr[:,1:], fr_len-1)
        en_msgs_ = self.fr_en.dec.beam(fr_hid, fr_len-1) # (5, batch_size, en_len)
        en_msgs, en_lens = [], []
        for en in en_msgs_:
            en = [ee[1:] for ee in en]
            en_len = [len(x) for x in en]
            max_len = max(en_len)
            en_len = cuda( torch.LongTensor( en_len ) )
            en_lens.append( en_len )

            en = [ np.lib.pad( xx, (0, max_len - len(xx)), 'constant', constant_values=(0,0) ) for xx in en ]
            en = cuda( torch.LongTensor( np.array(en) ) )
            en_msgs.append( en )

        en_hids = [ self.en_de.enc(en_msg, en_len) for (en_msg, en_len) in zip(en_msgs, en_lens) ]
        de_msg = self.en_de.dec.multi_decode(en_hids, en_lens)

        return en_msgs, de_msg

    def gumbel_forward(self, batch, en_lm=None, all_img=None, ranker=None):
        results = {}
        (fr, fr_len) = batch.fr
        (en, en_len) = batch.en
        (de, de_len) = batch.de
        batch_size = len(batch)

        # <BOS> removed from source Fr sentences when training Fr->En agent, hence fr_len - 1
        fr_hid = self.fr_en.enc(fr[:, 1:], fr_len-1)
        # Need to predict everything except <BOS>, hence en_len-1
        send_results = self.fr_en.dec.send(fr_hid, fr_len-1, en_len-1, "gumbel", self.value_fn, self.gumbel_temp)
        en_msg, en_msg_len = [send_results[key] for key in ["msg", "new_seq_lens"]]
        results.update(send_results)

        de_input, de_target = de[:, :-1], de[:, 1:].contiguous().view(-1)
        de_logits, _ = self.en_de(self.fr_en.gumbel_tokens, en_msg_len, de_input) # (batch_size * en_seq_len, vocab_size)
        de_nll = F.cross_entropy(de_logits, de_target, ignore_index=0, reduction='none')
        results['ce_loss'] = de_nll.mean()
        de_nll = de_nll.view(batch_size, -1).sum(dim=1) / (de_len - 1).float() # (batch_size)
        fr_en_results, fr_en_rewards = self.eval_fr_en_stats(en_msg, en_msg_len, batch, en_lm=en_lm,
                                                             all_img=all_img, ranker=ranker)
        results.update(fr_en_results)

        if not (self.fr_en.dec.neg_Hs is []):
            neg_Hs = self.fr_en.dec.neg_Hs # (batch_size, en_msg_len)
            neg_Hs = neg_Hs.mean() # (1,)
            results["neg_Hs"] = neg_Hs
        return results
