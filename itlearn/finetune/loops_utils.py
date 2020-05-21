"""
Utility for loop
"""
from tensorboardX import SummaryWriter
from pathlib import Path
import math
import numpy as np
import torch

from itlearn.utils.metrics import Metrics, Best
from itlearn.finetune.agents_utils import valid_model, eval_model
from itlearn.utils.misc import write_tb


class Trainer:
    def __init__(self, args, model, train_it, dev_it, extra_input,
                 loss_cos, loss_names, monitor_names):
        self.args = args
        self.model = model
        self.train_it = train_it
        self.dev_it = dev_it
        self.extra_input = extra_input
        self.loss_cos = loss_cos
        self.loss_names = loss_names
        self.monitor_names = monitor_names

        # Prepare writer
        self.writer = SummaryWriter( args.event_path + args.id_str)

        # Prepare opt
        if args.fix_fr2en:
            args.logger.info('Fix Fr En')
            self.params = [p for p in model.en_de.parameters() if p.requires_grad]
        else:
            self.params = [p for p in model.parameters() if p.requires_grad]
        if args.optimizer == 'Adam':
            self.opt = torch.optim.Adam(self.params, betas=(0.9, 0.98), eps=1e-9, lr=args.lr)
        else:
            raise NotImplementedError

        # Prepare S2P
        self.s2p_steps = args.__dict__.get('s2p_steps', args.max_training_steps)
        self.s2p_fr_en_it, self.s2p_en_de_it = None, None
        if hasattr(args, 's2p_freq') and args.s2p_freq > 0:
            args.logger.info('Perform S2P at every {} steps'.format(args.s2p_freq))
            fr_en_it, en_de_it = extra_input['s2p_its']['fr_en'], extra_input['s2p_its']['en_de']
            self.s2p_fr_en_it = iter(fr_en_it)
            self.s2p_en_de_it = iter(en_de_it)

    def joint_training(self):
        # Prepare Metrics
        train_metrics = Metrics('train_loss', *self.loss_names, *self.monitor_names, data_type="avg")
        best = Best(max, 'de_bleu', 'en_bleu', 'iters', model=self.model, opt=self.opt,
                    path=self.args.model_path + self.args.id_str,
                    gpu=self.args.gpu, debug=self.args.debug)

        for iters, train_batch in enumerate(self.train_it):
            if iters >= self.args.max_training_steps:
                self.args.logger.info('stopping training after {} training steps'.format(self.args.max_training_steps))
                break

            self._maybe_save(iters)

            if iters % self.args.eval_every == 0:
                self.eval_loop(iters, best)

            self.selfplay_step(iters, train_batch)

            if iters % self.args.eval_every == 0:
                self.args.logger.info("update {} : {}".format(iters, str(train_metrics)))
                write_tb(self.writer, self.loss_names,
                         [train_metrics.__getattr__(name) for name in self.loss_names],
                         iters, prefix="train/")
                write_tb(self.writer, self.monitor_names,
                         [train_metrics.__getattr__(name) for name in self.monitor_names],
                         iters, prefix="train/")
                write_tb(self.writer, ['lr'], [self.opt.param_groups[0]['lr']], iters, prefix="train/")
                train_metrics.reset()

                if self.args.plot_grad:
                    self._plot_grad(iters, train_batch)


    def _maybe_save(self, iters):
        if hasattr(self.args, 'save_every') and iters % self.args.save_every == 0:
            self.args.logger.info('save (back-up) checkpoints at iters={}'.format(iters))
            with torch.cuda.device(self.args.gpu):
                torch.save(self.model.state_dict(), '{}_iter={}.pt'.format(self.args.model_path + self.args.id_str,
                                                                           iters))
                torch.save([iters, self.opt.state_dict()],
                           '{}_iter={}.pt.states'.format(self.args.model_path + self.args.id_str, iters))

    def selfplay_step(self, iters, train_batch):
        """ Perform a step of selfplay """
        self.model.train()
        if hasattr(self.args, 'lr_anneal') and self.args.lr_anneal == "linear":
            self.opt.param_groups[0]['lr'] = self._get_lr_anneal(iters)
        if hasattr(self.args, 'h_co_anneal') and self.args.h_co_anneal == "linear":
            self.loss_cos['neg_Hs'] = self._get_h_co_anneal(iters)

        self.opt.zero_grad()
        batch_size = len(train_batch)
        R = self.model(train_batch, en_lm=self.extra_input["en_lm"],
                       all_img=self.extra_input["img"]['multi30k'][0],
                       ranker=self.extra_input["ranker"])
        losses = [R[key] for key in self.loss_names]
        total_loss = 0
        for loss_name, loss in zip(self.loss_names, losses):
            total_loss += loss * self.loss_cos[loss_name]
        self.train_metrics.accumulate(batch_size, *[loss.item() for loss in losses],
                                      *[R[k].item() for k in self.monitor_names])

        # Add S2P Grad
        if self.args.s2p_freq > 0 and iters % self.args.s2p_freq == 0:
            fr_en_loss, en_de_loss = self._s2p_batch()
            total_loss += self.args.s2p_co * (fr_en_loss + en_de_loss)

        total_loss.backward()
        if self.args.grad_clip > 0:
            total_norm = torch.nn.utils.clip_grad_norm_(self.params, self.args.grad_clip)
            if total_norm != total_norm or math.isnan(total_norm) or np.isnan(total_norm):
                print('NAN!!!!!!!!!!!!!!!!!!!!!!')
                exit()
        self.opt.step()

    def eval_loop(self, iters, best):
        dev_metrics = valid_model(self.model, self.dev_it, self.loss_names, self.monitor_names, self.extra_input)
        eval_metric, bleu_en, bleu_de, en_corpus, en_hyp, de_hyp = eval_model(self.args, self.model, self.dev_it,
                                                                              self.monitor_names, self.extra_input)
        write_tb(self.writer, self.loss_names, [dev_metrics.__getattr__(name) for name in self.loss_names],
                 iters, prefix="dev/")
        write_tb(self.writer, self.monitor_names, [dev_metrics.__getattr__(name) for name in self.monitor_names],
                 iters, prefix="dev/")
        write_tb(self.writer, ['bleu', *("p_1 p_2 p_3 p_4".split()), 'bp', 'len_ref', 'len_hyp'], bleu_en, iters,
                 prefix="bleu_en/")
        write_tb(self.writer, ['bleu', *("p_1 p_2 p_3 p_4".split()), 'bp', 'len_ref', 'len_hyp'], bleu_de, iters,
                 prefix="bleu_de/")
        write_tb(self.writer, ["bleu_en", "bleu_de"], [bleu_en[0], bleu_de[0]], iters, prefix="eval/")
        write_tb(self.writer, self.monitor_names, [eval_metric.__getattr__(name) for name in self.monitor_names],
                 iters, prefix="eval/")
        self.args.logger.info('model:' + self.args.prefix + self.args.hp_str)
        best.accumulate(bleu_de[0], bleu_en[0], iters)
        self.args.logger.info(best)

        # Save decoding results
        dest_folders = [Path(self.args.decoding_path) / self.args.id_str / name for name in
                        ["en_ref", "de_hyp_{}".format(iters), "en_hyp_{}".format(iters)]]
        for (dest, string) in zip(dest_folders, [en_corpus, de_hyp, en_hyp]):
            dest.write_text("\n".join(string), encoding="utf-8")

    def _get_lr_anneal(self, iters):
        lr_end = self.args.lr_min
        decay_lr = (self.args.lr - lr_end) * (self.args.linear_anneal_steps - iters) / self.args.linear_anneal_steps
        return max(0, decay_lr) + lr_end

    def _get_h_co_anneal(self, iters):
        h_co_end = self.args.h_co_min
        decay_h = (self.args.h_co - h_co_end) * (self.args.h_co_anneal_steps - iters) / self.args.h_co_anneal_steps
        return max(0, decay_h) + h_co_end

    def _plot_grad(self, iters, train_batch):
        """ plot the gradients for selfplau and supervise """
        self.model.train()
        self.opt.zero_grad()
        R = self.model(train_batch, en_lm=self.extra_input["en_lm"],
                       all_img=self.extra_input["img"]['multi30k'][0],
                       ranker=self.extra_input["ranker"])
        losses = [R[key] for key in self.loss_names]
        total_loss = 0
        for loss_name, loss in zip(self.loss_names, losses):
            assert loss.grad_fn is not None
            total_loss += loss * self.loss_cos[loss_name]
        total_loss.backward()
        rl_grad = torch.cat([p.grad.clone().reshape(-1) for p in self.params if p.grad is not None])
        self.opt.zero_grad()
        fr_en_loss, en_de_loss = self._s2p_batch()
        (fr_en_loss + en_de_loss).backward()
        s2p_grad = torch.cat([p.grad.clone().reshape(-1) for p in self.params if p.grad is not None])
        rl_grad_norm = torch.norm(rl_grad)
        s2p_grad_norm = torch.norm(s2p_grad)
        cosine = rl_grad.matmul(s2p_grad) / (rl_grad_norm * s2p_grad_norm)
        self.writer.add_scalar("grad/rl_grad_norm", rl_grad_norm.item(), global_step=iters)
        self.writer.add_scalar("grad/s2p_grad_norm", s2p_grad_norm.item(), global_step=iters)
        self.writer.add_scalar("grad/cosine", cosine.item(), global_step=iters)

    def _s2p_batch(self):
        fr_en_batch = self.s2p_fr_en_it.__next__()
        en_de_batch = self.s2p_en_de_it.__next__()
        fr_en_loss = _get_nll(self.model.fr_en,
                              fr_en_batch.src[0],
                              fr_en_batch.src[1],
                              fr_en_batch.trg[0])
        en_de_loss = _get_nll(self.model.en_de,
                              en_de_batch.src[0],
                              en_de_batch.src[1],
                              en_de_batch.trg[0])
        return fr_en_loss, en_de_loss


def _get_nll(model, src, src_len, trg):
    logits, _ = model(src[:, 1:], src_len - 1, trg[:, :-1])
    loss = torch.nn.functional.cross_entropy(logits, trg[:, 1:].contiguous().view(-1),
                                             reduction='mean', ignore_index=0)
    return loss



