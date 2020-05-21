import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from itlearn.models.agent import RNNAttn
from itlearn.utils.misc import cuda, xlen_to_inv_mask, sum_reward
from itlearn.models.modules import ArgsModule


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

        if hasattr(args, 'trim_dots') and args.trim_dots:
            args.logger.info('Trim Dots!')
            self.dot_token = args.EN.vocab.stoi['.']
        else:
            self.dot_token = None

    def fr_en_speak(self, batch, is_training=False):
        """ Different way for fr en speak """
        raise NotImplementedError

    def decode(self, batch):
        """ A helper for greedy decode """
        (fr, fr_len) = batch.fr
        (en, en_len) = batch.en
        (de, de_len) = batch.de
        fr_hid = self.fr_en.enc(fr[:, 1:], fr_len - 1)
        en_send_results = self.fr_en.dec.send(src_hid=fr_hid, src_len=fr_len - 1,
                                              trg_len=en_len - 1, send_method='argmax',
                                              dot_token=self.dot_token)
        en_msg, en_msg_len = [en_send_results[key] for key in ["msg", "new_seq_lens"]]
        en_hid = self.en_de.enc(en_msg, en_msg_len)
        de_send_results = self.en_de.dec.send(src_hid=en_hid, src_len=en_msg_len,
                                              trg_len=de_len - 1, send_method='argmax')
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

    def forward_fr_en(self, en_msg, en_msg_len, batch, en_lm=None, all_img=None, ranker=None,
                      use_gumbel_tokens=False):
        """ Forward speaker with English sentence to get loss and rewards """
        results = {}
        rewards = {}
        batch_size = en_msg.shape[0]
        # NOTE add <BOS> to beginning
        en_msg_ = torch.cat([cuda(torch.full((batch_size, 1), self.init_token)).long(), en_msg], dim=1)
        gumbel_tokens = None
        if use_gumbel_tokens:
            gumbel_tokens = self.fr_en.dec.gumbel_tokens
            init_tokens = torch.zeros([gumbel_tokens.shape[0], 1, gumbel_tokens.shape[2]])
            init_tokens = init_tokens.to(device=gumbel_tokens.device)
            init_tokens[:, :, self.init_token] = 1
            gumbel_tokens = torch.cat([init_tokens, gumbel_tokens], dim=1)

        if self.use_en_lm:  # monitor EN LM NLL
            if "wiki" in self.en_lm_dataset:
                if use_gumbel_tokens:
                    raise NotImplementedError
                en_nll_lm = en_lm.get_nll(en_msg_)  # (batch_size, en_msg_len)
                if self.train_en_lm:
                    en_nll_lm = sum_reward(en_nll_lm, en_msg_len + 1)  # (batch_size)
                    rewards['lm'] = -1 * en_nll_lm.detach()
                    # R = R + -1 * en_nll_lm.detach() * self.en_lm_nll_co # (batch_size)
                results.update({"en_nll_lm": en_nll_lm.mean()})

            elif self.en_lm_dataset in ["coco", "multi30k"]:
                if use_gumbel_tokens:
                    en_lm.train()
                    en_nll_lm = en_lm.get_loss_oh(gumbel_tokens, None)
                    en_lm.eval()
                else:
                    en_nll_lm = en_lm.get_loss(en_msg_, None)  # (batch_size, en_msg_len)
                if self.train_en_lm:
                    en_nll_lm = sum_reward(en_nll_lm, en_msg_len + 1)  # (batch_size)
                    rewards['lm'] = -1 * en_nll_lm.detach()
                results.update({"en_nll_lm": en_nll_lm.mean()})
            else:
                raise Exception()

        if self.use_ranker:  # NOTE Experiment 3 : Reward = NLL_DE + NLL_EN_LM + NLL_IMG_PRED
            if use_gumbel_tokens and self.train_ranker:
                raise NotImplementedError
            ranker.eval()
            img = cuda(all_img.index_select(dim=0, index=batch.idx.cpu()))  # (batch_size, D_img)

            if self.img_pred_loss == "nll":
                img_pred_loss = ranker.get_loss(en_msg_, img)  # (batch_size, en_msg_len)
                img_pred_loss = sum_reward(img_pred_loss, en_msg_len + 1)  # (batch_size)
            else:
                with torch.no_grad():
                    img_pred_loss = ranker(en_msg, en_msg_len, img)["loss"]

            if self.train_ranker:
                rewards['img_pred'] = -1 * img_pred_loss.detach()
            results.update({"img_pred_loss_{}".format(self.img_pred_loss): img_pred_loss.mean()})

            # Get ranker retrieval result
            with torch.no_grad():
                K = 19
                # Randomly select K distractor image
                random_idx = torch.randint(all_img.shape[0], size=[batch_size, K])
                wrong_img = cuda(all_img.index_select(dim=0, index=random_idx.view(-1)))
                wrong_img_feat = ranker.batch_enc_img(wrong_img).view(batch_size, K, -1)
                right_img_feat = ranker.batch_enc_img(img)

                # [bsz, K+1, hid_size]
                all_feat = torch.cat([right_img_feat.unsqueeze(1), wrong_img_feat], dim=1)

                # [bsz, hid_size]
                cap_feats = ranker.batch_cap_rep(en_msg, en_msg_len)
                scores = (cap_feats.unsqueeze(1) * all_feat).sum(-1)
                r1_acc = (torch.argmax(scores, -1) == 0).float().mean()
                results['r1_acc'] = r1_acc
        return results, rewards


class AgentsA2C(BaseAgents):
    def __init__(self, args):
        super(AgentsA2C, self).__init__(args)

    def fr_en_speak(self, batch, is_training=False):
        """ Speak with reinforce or greedy if not training """
        (fr, fr_len) = batch.fr
        (en, en_len) = batch.en
        fr_hid = self.fr_en.enc(fr[:, 1:], fr_len - 1)
        if is_training:
            send_results = self.fr_en.dec.send(src_hid=fr_hid, src_len=fr_len - 1,
                                               trg_len=en_len - 1, send_method="reinforce",
                                               value_fn=self.value_fn,
                                               dot_token=self.dot_token)
        else:
            send_results = self.fr_en.dec.send(src_hid=fr_hid, src_len=fr_len - 1,
                                               trg_len=en_len - 1, send_method='argmax',
                                               dot_token=self.dot_token)
        en_msg, en_msg_len = [send_results[key] for key in ["msg", "new_seq_lens"]]
        return en_msg, en_msg_len

    def forward(self, batch, en_lm=None, all_img=None, ranker=None):
        """ Return all stuff related to reinforce """
        results = {}
        rewards = {}

        # Speak fr en first
        en_msg, en_msg_len = self.fr_en_speak(batch, is_training=True)
        fr_en_results, fr_en_rewards = self.forward_fr_en(self, en_msg, en_msg_len, batch, en_lm=en_lm,
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
            send_results = self.fr_en.dec.send(src_hid=fr_hid, src_len=fr_len - 1,
                                               trg_len=en_len - 1, send_method="gumbel",
                                               gumbel_temp=self.gumbel_temp,
                                               dot_token=self.dot_token)
        else:
            send_results = self.fr_en.dec.send(src_hid=fr_hid, src_len=fr_len - 1,
                                               trg_len=en_len - 1, send_method='argmax',
                                               dot_token=self.dot_token)
        en_msg, en_msg_len = [send_results[key] for key in ["msg", "new_seq_lens"]]
        return en_msg, en_msg_len

    def forward(self, batch, en_lm=None, all_img=None, ranker=None):
        """ Create training graph """
        results = {}
        en_msg, en_msg_len = self.fr_en_speak(batch, is_training=True)
        fr_en_results, _ = self.forward_fr_en(self, en_msg, en_msg_len, batch, en_lm=en_lm,
                                              all_img=all_img, ranker=ranker, use_gumbel_tokens=self.training)
        results.update(fr_en_results)

        (de, de_len) = batch.de
        de_input, de_target = de[:, :-1], de[:, 1:].contiguous().view(-1)
        de_logits, _ = self.en_de(self.fr_en.dec.gumbel_tokens, en_msg_len, de_input) # (batch_size * en_seq_len, vocab_size)
        de_nll = F.cross_entropy(de_logits, de_target, ignore_index=0, reduction='none')
        results['ce_loss'] = de_nll.mean()

        # Get entropy
        neg_Hs = self.fr_en.dec.neg_Hs  # (batch_size, en_msg_len)
        neg_Hs = neg_Hs.mean()  # (1,)
        results["neg_Hs"] = neg_Hs
        return results
