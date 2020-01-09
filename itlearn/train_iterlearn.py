import ipdb
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from utils import token_analysis, get_counts, write_tb, plot_grad, cuda, xlen_to_inv_mask
from metrics import Metrics, Best
from misc.bleu import computeBLEU, compute_bp, print_bleu
from run_utils import get_model
import matplotlib.pyplot as plt
import os
from pathlib import Path


def eval_model(args, model, dev_it, monitor_names, iters, extra_input):
    """ Use greedy decoding and check scores like BLEU, language model and grounding """
    eval_metrics = Metrics('dev_loss', *monitor_names, data_type="avg")
    eval_metrics.reset()
    with torch.no_grad():
        unbpe = True
        model.eval()
        fr_corpus, en_corpus, de_corpus = [], [], []
        en_hyp, de_hyp = [], []

        for j, dev_batch in enumerate(dev_it):
            fr_corpus.extend(args.FR.reverse(dev_batch.fr[0], unbpe=unbpe))
            en_corpus.extend(args.EN.reverse(dev_batch.en[0], unbpe=unbpe))
            de_corpus.extend(args.DE.reverse(dev_batch.de[0], unbpe=unbpe))

            en_msg, de_msg, en_msg_len, _ = model.decode(dev_batch)
            en_hyp.extend(args.EN.reverse(en_msg, unbpe=unbpe))
            de_hyp.extend(args.DE.reverse(de_msg, unbpe=unbpe))
            results, _ = model.eval_fr_en_stats(en_msg, en_msg_len, dev_batch,
                                                en_lm=extra_input["en_lm"],
                                                all_img=extra_input["img"]['multi30k'][1],
                                                ranker=extra_input["ranker"])
            if len(monitor_names) > 0:
                eval_metrics.accumulate(len(dev_batch), *[results[k].item() for k in monitor_names])

        bleu_en = computeBLEU(en_hyp, en_corpus, corpus=True)
        bleu_de = computeBLEU(de_hyp, de_corpus, corpus=True)
        args.logger.info(eval_metrics)
        args.logger.info("Fr-En {} : {}".format('valid', print_bleu(bleu_en)))
        args.logger.info("En-De {} : {}".format('valid', print_bleu(bleu_de)))

        if not args.debug:
            dest_folders = [Path(args.decoding_path) / args.id_str / name for name in
                            ["en_ref", "de_ref", "fr_ref", "de_hyp_{}".format(iters), "en_hyp_{}".format(iters)]]
            [dest.write_text("\n".join(string), encoding="utf-8")
             for (dest, string) in zip(dest_folders, [en_corpus, de_corpus, fr_corpus, de_hyp, en_hyp])]
        return eval_metrics, bleu_en, bleu_de


def valid_model(model, dev_it, loss_names, monitor_names, extra_input):
    """ Run reinforce on validation and record stats """
    dev_metrics = Metrics('dev_loss', *loss_names, *monitor_names, data_type="avg")
    dev_metrics.reset()
    with torch.no_grad():
        model.eval()
        for j, dev_batch in enumerate(dev_it):
            R = model(dev_batch, en_lm=extra_input["en_lm"], all_img=extra_input["img"]['multi30k'][1],
                      ranker=extra_input["ranker"])
            losses = [R[key] for key in loss_names]
            dev_metrics.accumulate(len(dev_batch), *[loss.item() for loss in losses],
                                   *[R[k].item() for k in monitor_names])
    return dev_metrics


def get_lr_anneal(args, iters):
    lr_end = args.lr_min
    return max(0, (args.lr - lr_end) * (args.linear_anneal_steps - iters) / args.linear_anneal_steps) + lr_end


def get_h_co_anneal(args, iters):
    h_co_end = args.h_co_min
    return max(0, (args.h_co - h_co_end) * (args.h_co_anneal_steps - iters) / args.h_co_anneal_steps) + h_co_end


def selfplay_step(args, extra_input, iters, loss_cos, loss_names, model, monitor_names, opt, params, train_batch,
                  train_metrics, writer):
    """ Perform a step of selfplay """
    if args.lr_anneal == "linear":
        opt.param_groups[0]['lr'] = get_lr_anneal(args, iters)
    if args.h_co_anneal == "linear":
        loss_cos['neg_Hs'] = get_h_co_anneal(args, iters)
    opt.zero_grad()
    batch_size = len(train_batch)
    R = model(train_batch, en_lm=extra_input["en_lm"], all_img=extra_input["img"]['multi30k'][0],
              ranker=extra_input["ranker"])
    losses = [R[key] for key in loss_names]
    total_loss = 0
    for loss_name, loss in zip(loss_names, losses):
        total_loss += loss * loss_cos[loss_name]
    train_metrics.accumulate(batch_size, *[loss.item() for loss in losses], *[R[k].item() for k in monitor_names])
    total_loss.backward()
    if args.plot_grad:
        plot_grad(writer, model, iters)
    if args.grad_clip > 0:
        total_norm = nn.utils.clip_grad_norm_(params, args.grad_clip)
        if total_norm != total_norm or math.isnan(total_norm) or np.isnan(total_norm):
            ipdb.set_trace()
    opt.step()

    if iters % args.eval_every == 0:
        args.logger.info("update {} : {}".format(iters, str(train_metrics)))
        if iters % args.eval_every == 0 and not args.debug:
            write_tb(writer, loss_names, [train_metrics.__getattr__(name) for name in loss_names],
                     iters, prefix="train/")
            write_tb(writer, monitor_names, [train_metrics.__getattr__(name) for name in monitor_names],
                     iters, prefix="train/")
            write_tb(writer, ['lr'], [opt.param_groups[0]['lr']], iters, prefix="train/")
            if not args.fix_fr2en:
                write_tb(writer, ["h_co"], [loss_cos['neg_Hs']], iters, prefix="train/")
            train_metrics.reset()


def train_model(args, teacher, iterators, extra_input):
    (train_it, dev_it) = iterators

    if not args.debug:
        decoding_path = Path(args.decoding_path + args.id_str)
        decoding_path.mkdir(parents=True, exist_ok=True)
        from tensorboardX import SummaryWriter
        writer = SummaryWriter( args.event_path + args.id_str)

    params = [p for p in teacher.parameters() if p.requires_grad]
    if args.optimizer == 'Adam':
        opt = torch.optim.Adam(params, betas=(0.9, 0.98), eps=1e-9, lr=args.lr)
    else:
        raise NotImplementedError

    monitor_names = []
    loss_names = ['ce_loss']
    loss_cos = {"ce_loss": args.ce_co}
    if not args.fix_fr2en:
        loss_names.extend(['pg_loss', 'b_loss'])
        loss_cos.update({'pg_loss': args.pg_co, 'b_loss': args.b_co} )

        if args.h_co > 0:
            loss_names.append('neg_Hs')
        else:
            monitor_names.append('neg_Hs')
        loss_cos['neg_Hs'] = args.h_co

    if args.use_ranker:
        monitor_names.extend(["img_pred_loss_{}".format(args.img_pred_loss)])
    if args.use_en_lm:
        monitor_names.append('en_nll_lm')

    train_metrics = Metrics('train_loss', *loss_names, *monitor_names, data_type="avg")
    best = Best(max, 'de_bleu', 'en_bleu', 'iters', model=teacher, opt=opt, path=args.model_path + args.id_str, gpu=args.gpu, debug=args.debug)

    # Prepare_init_student
    student = get_model(args)
    student.load_state_dict(teacher.state_dict())
    if torch.cuda.is_available() and args.gpu > -1:
        student.cuda(args.gpu)
    for iters, train_batch in enumerate(train_it):
        if iters >= args.max_training_steps:
            args.logger.info('stopping training after {} training steps'.format(args.max_training_steps))
            break

        if not args.debug and iters in args.save_at:
            args.logger.info('save (back-up) checkpoints at iters={}'.format(iters))
            with torch.cuda.device(args.gpu):
                torch.save(teacher.state_dict(), '{}_iter={}.pt'.format(args.model_path + args.id_str, iters))
                torch.save([iters, opt.state_dict()], '{}_iter={}.pt.states'.format( args.model_path + args.id_str, iters))

        if iters % args.eval_every == 0:
            dev_metrics = valid_model(teacher, dev_it, loss_names, monitor_names, extra_input)
            eval_metric, bleu_en, bleu_de = eval_model(args, teacher, dev_it, monitor_names, iters, extra_input)
            if not args.debug:
                write_tb(writer, loss_names, [dev_metrics.__getattr__(name) for name in loss_names],
                         iters, prefix="dev/")
                write_tb(writer, monitor_names, [dev_metrics.__getattr__(name) for name in monitor_names],
                         iters, prefix="dev/")

                write_tb(writer, ['bleu', *("p_1 p_2 p_3 p_4".split()), 'bp', 'len_ref', 'len_hyp'], bleu_en, iters,
                         prefix="bleu_en/")
                write_tb(writer, ['bleu', *("p_1 p_2 p_3 p_4".split()), 'bp', 'len_ref', 'len_hyp'], bleu_de, iters,
                         prefix="bleu_de/")
                write_tb(writer, ["bleu_en", "bleu_de"], [bleu_en[0], bleu_de[0]], iters, prefix="eval/")
                write_tb(writer, monitor_names, [eval_metric.__getattr__(name) for name in monitor_names],
                         iters, prefix="eval/")

            args.logger.info('model:' + args.prefix + args.hp_str)
            best.accumulate(bleu_de[0], bleu_en[0], iters)
            args.logger.info(best)
            if args.early_stop and (iters - best.iters) // args.eval_every > args.patience:
                args.logger.info("Early stopping.")
                break

        teacher.train()
        selfplay_step(args, extra_input, iters, loss_cos, loss_names, teacher, monitor_names, opt, params, train_batch,
                      train_metrics, writer)

        if (iters + 1) % args.generation_steps == 0:
            args.logger.info('start imitating...')
            student.train()
            teacher.eval()
            imitate_statss = imitate(args, student, teacher, train_it, dev_it, monitor_names, extra_input)
            teacher.load_state_dict(student.state_dict())
            if args.save_imitate_stats:
                teacher_stats = get_imitate_stats(args, teacher, dev_it, monitor_names, extra_input)
                iterss = [res[0] for res in imitate_statss]
                statss = [res[1] for res in imitate_statss]
                fig, axs = plt.subplots(len(teacher_stats), figsize=(7, 7*len(teacher_stats)))
                for name, ax in zip(teacher_stats, axs.reshape(-1)):
                    student_vals = [stats[name] for stats in statss]
                    teacher_val = teacher_stats[name]
                    ax.plot(iterss, student_vals, label='student')
                    ax.plot([iterss[0], iterss[-1]], [teacher_val, teacher_val], label='teacher')
                    ax.set_title(name)
                    ax.legend()
                print('Save imitation stats')
                if not os.path.exists(os.path.join(args.misc_path, args.id_str)):
                    os.makedirs(os.path.join(args.misc_path, args.id_str))
                fig.savefig(os.path.join(args.misc_path, args.id_str, 'imitation_{}_stats.png'.format(iters + 1)))
                del fig


def imitate(args, student_models, teacher_models, train_it, dev_it, monitor_names, extra_input):
    s_model, l_model = student_models.fr_en, student_models.en_de
    s_params = [p for p in s_model.parameters() if p.requires_grad]
    s_opt = torch.optim.Adam(s_params, betas=(0.9, 0.98), eps=1e-9, lr=args.s_lr)
    l_params = [p for p in l_model.parameters() if p.requires_grad]
    l_opt = torch.optim.Adam(s_params, betas=(0.9, 0.98), eps=1e-9, lr=args.l_lr)
    imitate_statss = []
    for iters, batch in enumerate(train_it):
        if iters >= args.learn_steps:
            args.logger.info('student stop learning after {} training steps'.format(args.learn_steps))
            break

        if args.save_imitate_stats and iters % 10 == 0:
            args.logger.info('Record imitate stats at {}'.format(iters))
            student_models.eval()
            stats = get_imitate_stats(args, student_models, dev_it, monitor_names, extra_input)
            imitate_statss.append((iters, stats))

        # Teacher generate message
        with torch.no_grad():
            en_msg, de_msg, en_msg_len, de_msg_len = teacher_models.decode(batch)
            en_msg, en_msg_len = _make_sure_message_valid(en_msg, en_msg_len, teacher_models.init_token)
            de_msg, de_msg_len = _make_sure_message_valid(de_msg, de_msg_len, teacher_models.init_token)

        student_models.train()

        # Get fr en
        speaker_nll = _get_nll(student_models.fr_en, batch.fr[0], batch.fr[1], en_msg)
        s_opt.zero_grad()
        speaker_nll.backward()
        nn.utils.clip_grad_norm_(s_params, 0.1)
        s_opt.step()

        # Get en de
        listener_nll = _get_nll(student_models.en_de, en_msg, en_msg_len, de_msg)
        l_opt.zero_grad()
        listener_nll.backward()
        nn.utils.clip_grad_norm_(l_params, 0.1)
        l_opt.step()

    return imitate_statss


def _make_sure_message_valid(msg, msg_len, init_token):
    # Add BOS
    msg = torch.cat([cuda(torch.full((msg.shape[0], 1), init_token)).long(), msg],
                    dim=1)
    msg_len += 1

    # Make sure padding are all zeros
    inv_mask = xlen_to_inv_mask(msg_len, seq_len=msg.shape[1])
    msg.masked_fill_(mask=inv_mask.bool(), value=0)
    return msg, msg_len


def _get_nll(single_model, src, src_len, trg):
    """ Single model get NLL loss"""
    # NOTE encoder never receives <BOS> token
    # because during communication, Agent A's decoder will never output <BOS>
    logits, _ = single_model(src[:, 1:], src_len - 1, trg[:, :-1])
    nll = F.cross_entropy(logits, trg[:, 1:].contiguous().view(-1), reduction='mean',
                          ignore_index=0)
    return nll


def get_imitate_stats(args, model, dev_it, monitor_names, extra_input):
    """ Use greedy decoding and check scores like BLEU, language model and grounding """
    eval_metrics = Metrics('dev_loss', *monitor_names, data_type="avg")
    eval_metrics.reset()
    with torch.no_grad():
        unbpe = True
        model.eval()
        fr_corpus, en_corpus, de_corpus = [], [], []
        en_hyp, de_hyp = [], []

        for j, dev_batch in enumerate(dev_it):
            fr_corpus.extend(args.FR.reverse(dev_batch.fr[0], unbpe=unbpe))
            en_corpus.extend(args.EN.reverse(dev_batch.en[0], unbpe=unbpe))
            de_corpus.extend(args.DE.reverse(dev_batch.de[0], unbpe=unbpe))

            en_msg, de_msg, en_msg_len, _ = model.decode(dev_batch)
            en_hyp.extend(args.EN.reverse(en_msg, unbpe=unbpe))
            de_hyp.extend(args.DE.reverse(de_msg, unbpe=unbpe))
            results, _ = model.eval_fr_en_stats(en_msg, en_msg_len, dev_batch,
                                                en_lm=extra_input["en_lm"],
                                                all_img=extra_input["img"]['multi30k'][1],
                                                ranker=extra_input["ranker"])
            if len(monitor_names) > 0:
                eval_metrics.accumulate(len(dev_batch), *[results[k].item() for k in monitor_names])

        bleu_en = computeBLEU(en_hyp, en_corpus, corpus=True)
        bleu_de = computeBLEU(de_hyp, de_corpus, corpus=True)
        stats = eval_metrics.__dict__['metrics']
        stats['bleu_en'] = bleu_en[0]
        stats['bleu_de'] = bleu_de[0]
        return stats
