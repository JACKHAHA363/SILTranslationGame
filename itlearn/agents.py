import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from agent import RNNAttn
from agents_utils import eval_fr_en_stats
from utils import cuda, xlen_to_inv_mask
from modules import ArgsModule


class BaseAgents(ArgsModule):
    """ Base agents for both a2c and gumbel """
    def __init__(self, args):
        super(BaseAgents, self).__init__(args)
        rnncls = RNNAttn
        self.fr_en = rnncls(args, len(args.FR.vocab.itos), len(args.EN.vocab.itos))
        self.en_de = rnncls(args, len(args.EN.vocab.itos), len(args.DE.vocab.itos))
        self.en_de.dec.msg_len_ratio = -1.0
        value_fn_input_dim = args.D_hid + args.D_emb
        self.value_fn = nn.Sequential(
            nn.Linear(value_fn_input_dim, args.D_hid),
            nn.ReLU(),
            nn.Linear(args.D_hid, 1)
        )

    def fr_en_speak(self, batch, is_training=False):
        """ Different way for fr en speak """
        raise NotImplementedError

    def decode(self, batch):
        """ A helper for greedy decode """
        (fr, fr_len) = batch.fr
        (en, en_len) = batch.en
        (de, de_len) = batch.de
        fr_hid = self.fr_en.enc(fr[:, 1:], fr_len - 1)
        en_send_results = self.fr_en.dec.send(fr_hid, fr_len - 1, en_len - 1, 'argmax')
        en_msg, en_msg_len = [en_send_results[key] for key in ["msg", "new_seq_lens"]]
        en_hid = self.en_de.enc(en_msg, en_msg_len)
        de_send_results = self.en_de.dec.send(en_hid, en_msg_len, de_len - 1, 'argmax')
        de_msg, de_msg_len = [de_send_results[key] for key in ["msg", "new_seq_lens"]]
        return en_msg, de_msg, en_msg_len, de_msg_len

    def multi_decode(self, batch):
        """ A helper for beam """
        (fr, fr_len) = batch.fr
        (en, en_len) = batch.en
        (de, de_len) = batch.de
        batch_size = len(batch)

        fr_hid = self.fr_en.enc(fr[:, 1:], fr_len - 1)
        en_msgs_ = self.fr_en.dec.beam(fr_hid, fr_len - 1)  # (5, batch_size, en_len)
        en_msgs, en_lens = [], []
        for en in en_msgs_:
            en = [ee[1:] for ee in en]
            en_len = [len(x) for x in en]
            max_len = max(en_len)
            en_len = cuda(torch.LongTensor(en_len))
            en_lens.append(en_len)

            en = [np.lib.pad(xx, (0, max_len - len(xx)), 'constant', constant_values=(0, 0)) for xx in en]
            en = cuda(torch.LongTensor(np.array(en)))
            en_msgs.append(en)

        en_hids = [self.en_de.enc(en_msg, en_len) for (en_msg, en_len) in zip(en_msgs, en_lens)]
        de_msg = self.en_de.dec.multi_decode(en_hids, en_lens)

        return en_msgs, de_msg


class AgentsA2C(BaseAgents):
    def __init__(self, args):
        super(AgentsA2C, self).__init__(args)

    def fr_en_speak(self, batch, is_training=False):
        """ Speak with reinforce or greedy if not training """
        (fr, fr_len) = batch.fr
        (en, en_len) = batch.en
        fr_hid = self.fr_en.enc(fr[:, 1:], fr_len - 1)
        if is_training:
            send_results = self.fr_en.dec.send(fr_hid, fr_len - 1, en_len - 1, "reinforce", self.value_fn)
        else:
            send_results = self.fr_en.dec.send(fr_hid, fr_len - 1, en_len - 1, 'argmax')
        en_msg, en_msg_len = [send_results[key] for key in ["msg", "new_seq_lens"]]
        return en_msg, en_msg_len

    def forward(self, batch, en_lm=None, all_img=None, ranker=None):
        """ Return all stuff related to reinforce """
        results = {}
        rewards = {}

        # Speak fr en first
        en_msg, en_msg_len = self.fr_en_speak(batch)
        fr_en_results, fr_en_rewards = eval_fr_en_stats(self, en_msg, en_msg_len, batch, en_lm=en_lm,
                                                        all_img=all_img, ranker=ranker)
        results.update(fr_en_results)
        rewards.update(fr_en_rewards)

        # Speak De and get reward
        (de, de_len) = batch.de
        batch_size = len(batch)
        de_input, de_target = de[:, :-1], de[:, 1:].contiguous().view(-1)
        de_logits, _ = self.en_de(en_msg, en_msg_len, de_input) # (batch_size * en_seq_len, vocab_size)
        de_nll = F.cross_entropy(de_logits, de_target, ignore_index=0, reduction='none')
        results['ce_loss'] = de_nll.mean()
        de_nll = de_nll.view(batch_size, -1).sum(dim=1) / (de_len - 1).float() # (batch_size)
        rewards['ce'] = -1 * de_nll.detach() # NOTE Experiment 1 : Reward = NLL_DE

        # Entropy
        if not (self.fr_en.dec.neg_Hs is []):
            neg_Hs = self.fr_en.dec.neg_Hs # (batch_size, en_msg_len)
            neg_Hs = neg_Hs.mean() # (1,)
            results["neg_Hs"] = neg_Hs

        # Reward shaping
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


class AgentsGumbel(BaseAgents):
    def __init__(self, args):
        assert hasattr(args, 'gumbel_temp')
        super(AgentsGumbel, self).__init__(args)

    def fr_en_speak(self, batch, is_training=False):
        """ Fr en speak with gumbel """
        (fr, fr_len) = batch.fr
        (_, en_len) = batch.en
        fr_hid = self.fr_en.enc(fr[:, 1:], fr_len - 1)
        if is_training:
            send_results = self.fr_en.dec.send(fr_hid, fr_len - 1, en_len - 1, "gumbel",
                                               None, self.gumbel_temp)
        else:
            send_results = self.fr_en.dec.send(fr_hid, fr_len - 1, en_len - 1, 'argmax')
        en_msg, en_msg_len = [send_results[key] for key in ["msg", "new_seq_lens"]]
        return en_msg, en_msg_len

    def forward(self, batch, en_lm=None, all_img=None, ranker=None):
        """ Create training graph """
        results = {}
        en_msg, en_msg_len = self.fr_en_speak(batch)
        fr_en_results, _ = eval_fr_en_stats(self, en_msg, en_msg_len, batch, en_lm=en_lm,
                                            all_img=all_img, ranker=ranker)
        results.update(fr_en_results)

        (de, de_len) = batch.de
        de_input, de_target = de[:, :-1], de[:, 1:].contiguous().view(-1)
        de_logits, _ = self.en_de(self.fr_en.dec.gumbel_tokens, en_msg_len, de_input) # (batch_size * en_seq_len, vocab_size)
        de_nll = F.cross_entropy(de_logits, de_target, ignore_index=0, reduction='none')
        results['ce_loss'] = de_nll.mean()
        return results
