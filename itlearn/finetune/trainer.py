"""
Trainers
"""
from tensorboardX import SummaryWriter
from pathlib import Path
import math
import numpy as np
import torch

from itlearn.utils.bleu import computeBLEU, print_bleu
from itlearn.utils.metrics import Metrics, Best
from itlearn.utils.misc import write_tb


def _get_nll(model, src, src_len, trg):
    logits, _ = model(src[:, 1:], src_len - 1, trg[:, :-1])
    loss = torch.nn.functional.cross_entropy(logits, trg[:, 1:].contiguous().view(-1),
                                             reduction='mean', ignore_index=0)
    return loss


class Trainer:
    def __init__(self, args, model, train_it, dev_it, extra_input):
        self.args = args
        self.model = model
        self.train_it = train_it
        self.dev_it = dev_it
        self.extra_input = extra_input

        # Prepare loss
        self.loss_cos = {"ce_loss": args.ce_co}
        agents_type = model.__class__.__name__
        if agents_type == 'AgentsA2C':
            args.logger.info('Train with A2C')
            self.loss_cos.update({'pg_loss': args.pg_co, 'b_loss': args.b_co, 'neg_Hs': args.h_co})
        elif agents_type == 'AgentsGumbel':
            args.logger.info('Train with Gumbel')
            args.logger.info("Don't train entropy but observe it")
            self.loss_cos.update({'neg_Hs': 0.})

            # Use LM reward
            if args.train_en_lm:
                args.logger.info('Train with LM reward {}'.format(args.en_lm_nll_co))
                self.loss_cos['en_nll_lm'] = args.en_lm_nll_co

                # Use entropy coef as well. If KL then h_co = en_lm_nll_co
                args.logger.info('Train entropy with {}'.format(args.h_co))
                self.loss_cos['neg_Hs'] = args.h_co

            if args.train_ranker:
                img_pred_loss_name = "img_pred_loss_{}".format(args.img_pred_loss)
                args.logger.info('Train with grounding reward')
                self.loss_cos[img_pred_loss_name] = args.img_pred_loss_co
        else:
            raise ValueError
        args.logger.info('--------------- Loss -----------------')
        for name, coef in self.loss_cos.items():
            args.logger.info('{}: {}'.format(name, coef))
        args.logger.info('--------------------------------------')

        # Prepare monitor names
        self.monitor_names = ["neg_Hs"]
        if args.use_ranker:
            self.monitor_names.extend(["img_pred_loss_{}".format(args.img_pred_loss)])
            self.monitor_names.extend(["r1_acc"])
        if args.use_en_lm:
            self.monitor_names.append('en_nll_lm')
        args.logger.info('------------Monitor---------------')
        for monitor_name in self.monitor_names:
            args.logger.info(monitor_name)
        args.logger.info('----------------------------------')

        # Prepare writer
        self.writer = SummaryWriter( args.event_path + args.id_str)
        self.resume = 'resume' in extra_input

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
        if self.resume:
            self.opt.load_state_dict(extra_input['resume']['opt'])

        # Prepare S2P
        self.use_s2p = hasattr(args, 's2p_freq') and args.s2p_freq > 0
        self.s2p_steps = args.__dict__.get('s2p_steps', args.max_training_steps)
        if self.use_s2p:
            args.logger.info('Perform S2P at every {} steps'.format(args.s2p_freq))
            fr_en_it, en_de_it = extra_input['s2p_its']['fr_en'], extra_input['s2p_its']['en_de']
            self.s2p_fr_en_it = iter(fr_en_it)
            self.s2p_en_de_it = iter(en_de_it)

        # Make decoding path
        decoding_path = Path(args.decoding_path + args.id_str)
        decoding_path.mkdir(parents=True, exist_ok=True)

    def start(self):
        # Prepare Metrics
        train_metrics = Metrics('train_loss', *list(self.loss_cos.keys()), *self.monitor_names, data_type="avg")
        best = Best(max, 'de_bleu', 'en_bleu', 'iters', model=self.model, opt=self.opt,
                    path=self.args.model_path + self.args.id_str,
                    gpu=self.args.gpu, debug=self.args.debug)
        # Determine when to stop iterlearn
        iters = self.extra_input['resume']['iters'] if self.resume else 0
        self.args.logger.info('Start Training at iters={}'.format(iters))
        try:
            train_it = iter(self.train_it)
            while iters < self.args.max_training_steps:
                train_batch = train_it.__next__()
                if iters >= self.args.max_training_steps:
                    self.args.logger.info('stopping training after {} training steps'.format(self.args.max_training_steps))
                    break

                self._maybe_save(iters)

                if iters % self.args.eval_every == 0:
                    self.model.eval()
                    self.evaluate(iters, best)

                self.model.train()
                self.train_step(iters, train_batch, train_metrics)

                if iters % self.args.eval_every == 0:
                    self.args.logger.info("update {} : {}".format(iters, str(train_metrics)))
                    train_stats = {}
                    train_stats.update({name: train_metrics.__getattr__(name) for name in self.loss_cos})
                    train_stats.update({name: train_metrics.__getattr__(name) for name in self.monitor_names})
                    train_stats['lr'] = self.opt.param_groups[0]['lr']
                    write_tb(self.writer, train_stats, iters, prefix="train/")
                    train_metrics.reset()

                    if self.args.plot_grad:
                        self.model.train()
                        self._plot_grad(iters, train_batch)

                iters += 1
        except (InterruptedError, KeyboardInterrupt):
            # End Gracefully
            self.end_gracefully(iters)

    def train_step(self, iters, train_batch, train_metrics):
        """ Perform a step of selfplay as well as supervise loss if necessary """
        if hasattr(self.args, 'lr_anneal') and self.args.lr_anneal == "linear":
            self.opt.param_groups[0]['lr'] = self._get_lr_anneal(iters)
        if hasattr(self.args, 'h_co_anneal') and self.args.h_co_anneal == "linear":
            self.loss_cos['neg_Hs'] = self._get_h_co_anneal(iters)

        self.opt.zero_grad()
        batch_size = len(train_batch)
        train_result = self.model(train_batch, en_lm=self.extra_input["en_lm"],
                                  all_img=self.extra_input["img"]['multi30k'][0],
                                  ranker=self.extra_input["ranker"])
        total_loss = 0
        for loss_name, loss_co in self.loss_cos.items():
            if loss_co == 0:
                continue
            total_loss += loss_co * train_result[loss_name]
        train_metrics.accumulate(batch_size, **train_result)

        # Add S2P Grad
        if self.use_s2p and iters % self.args.s2p_freq == 0:
            fr_en_loss, en_de_loss = self._s2p_batch()
            total_loss += self.args.s2p_co * (fr_en_loss + en_de_loss)

        total_loss.backward()
        if self.args.grad_clip > 0:
            total_norm = torch.nn.utils.clip_grad_norm_(self.params, self.args.grad_clip)
            if total_norm != total_norm or math.isnan(total_norm) or np.isnan(total_norm):
                print('NAN!!!!!!!!!!!!!!!!!!!!!!')
                exit()
        self.opt.step()

    def end_gracefully(self, iters):
        self.args.logger.info('Interrupted! save (back-up) checkpoints at iters={}'.format(iters))
        self.writer.flush()
        self.writer.close()
        with torch.cuda.device(self.args.gpu):
            status = {'iters': iters,
                      'model': self.model.state_dict(),
                      'opt': self.opt.state_dict()}
            torch.save(status, '{}_latest.pt'.format(self.args.model_path + self.args.id_str))

    def evaluate_communication(self):
        """ Use greedy decoding and check scores like BLEU, language model and grounding """
        eval_metrics = Metrics('dev_loss', *self.monitor_names, data_type="avg")
        eval_metrics.reset()
        with torch.no_grad():
            unbpe = True
            self.model.eval()
            en_corpus, de_corpus = [], []
            en_hyp, de_hyp = [], []

            for j, dev_batch in enumerate(self.dev_it):
                en_corpus.extend(self.args.EN.reverse(dev_batch.en[0], unbpe=unbpe))
                de_corpus.extend(self.args.DE.reverse(dev_batch.de[0], unbpe=unbpe))

                en_msg, de_msg, en_msg_len, _ = self.model.decode(dev_batch)
                en_hyp.extend(self.args.EN.reverse(en_msg, unbpe=unbpe))
                de_hyp.extend(self.args.DE.reverse(de_msg, unbpe=unbpe))
                results, _ = self.model.get_grounding(en_msg, en_msg_len, dev_batch,
                                                      en_lm=self.extra_input["en_lm"],
                                                      all_img=self.extra_input["img"]['multi30k'][1],
                                                      ranker=self.extra_input["ranker"])
                # Get entropy
                neg_Hs = self.model.fr_en.dec.neg_Hs  # (batch_size, en_msg_len)
                neg_Hs = neg_Hs.mean()  # (1,)
                results["neg_Hs"] = neg_Hs
                if len(self.monitor_names) > 0:
                    eval_metrics.accumulate(len(dev_batch), **results)

            bleu_en = computeBLEU(en_hyp, en_corpus, corpus=True)
            bleu_de = computeBLEU(de_hyp, de_corpus, corpus=True)
            self.args.logger.info(eval_metrics)
            self.args.logger.info("Fr-En {} : {}".format('valid', print_bleu(bleu_en)))
            self.args.logger.info("En-De {} : {}".format('valid', print_bleu(bleu_de)))
            return eval_metrics, bleu_en, bleu_de, en_corpus, en_hyp, de_hyp

    def evaluate(self, iters, best):
        """ Free Run """
        eval_metric, bleu_en, bleu_de, en_corpus, en_hyp, de_hyp = self.evaluate_communication()
        bleu_names = ['bleu', "p_1", "p_2", "p_3", "p_4", 'bp', 'len_ref', 'len_hyp']
        write_tb(self.writer, {name: val for name, val in zip(bleu_names, bleu_en)}, iters, prefix='bleu_en/')
        write_tb(self.writer, {name: val for name, val in zip(bleu_names, bleu_de)}, iters, prefix='bleu_de/')
        write_tb(self.writer, {"bleu_en": bleu_en[0], "bleu_de": bleu_de[0]}, iters, prefix="eval/")
        write_tb(self.writer, {name: eval_metric.__getattr__(name) for name in self.monitor_names},
                 iters, prefix="eval/")
        self.args.logger.info('model:' + self.args.prefix + self.args.hp_str)
        best.accumulate(bleu_de[0], bleu_en[0], iters)
        self.args.logger.info(best)

        # Save decoding results
        dest_folders = [Path(self.args.decoding_path) / self.args.id_str / name for name in
                        ["en_ref", "de_hyp_{}".format(iters), "en_hyp_{}".format(iters)]]
        for (dest, string) in zip(dest_folders, [en_corpus, de_hyp, en_hyp]):
            dest.write_text("\n".join(string), encoding="utf-8")

    def _maybe_save(self, iters):
        if hasattr(self.args, 'save_every') and iters % self.args.save_every == 0:
            self.args.logger.info('save (back-up) checkpoints at iters={}'.format(iters))
            with torch.cuda.device(self.args.gpu):
                torch.save(self.model.state_dict(), '{}_iter={}.pt'.format(self.args.model_path + self.args.id_str,
                                                                           iters))
                torch.save([iters, self.opt.state_dict()],
                           '{}_iter={}.pt.states'.format(self.args.model_path + self.args.id_str, iters))

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
        assert self.use_s2p
        self.opt.zero_grad()
        fwd_results = self.model(train_batch, en_lm=self.extra_input["en_lm"],
                                 all_img=self.extra_input["img"]['multi30k'][0],
                                 ranker=self.extra_input["ranker"])
        total_loss = 0
        for name, co in self.loss_cos.items():
            if co == 0:
                continue
            assert fwd_results[name].grad_fn is not None
            total_loss += co * fwd_results[name]
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
