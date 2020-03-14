import math

import numpy as np
import torch
import torch.nn as nn

from utils import write_tb, plot_grad
from metrics import Metrics, Best
from run_utils import get_model
import matplotlib.pyplot as plt
from agents_utils import eval_model, valid_model
import os
from pandas import DataFrame
from pathlib import Path
from imitate_utils import imitate_fr_en, imitate_en_de, get_fr_en_imitate_stats, \
    get_en_de_imitate_stats, finetune_en_de


def get_lr_anneal(args, iters):
    lr_end = args.lr_min
    return max(0, (args.lr - lr_end) * (args.linear_anneal_steps - iters) / args.linear_anneal_steps) + lr_end


def get_h_co_anneal(args, iters):
    h_co_end = args.h_co_min
    return max(0, (args.h_co - h_co_end) * (args.h_co_anneal_steps - iters) / args.h_co_anneal_steps) + h_co_end


def selfplay_step(args, extra_input, iters, loss_cos, loss_names, model, monitor_names, opt, params, train_batch,
                  train_metrics, writer, s2p_fr_en_it, s2p_en_de_it):
    """ Perform a step of selfplay """
    if hasattr(args, 'lr_anneal') and args.lr_anneal == "linear":
        opt.param_groups[0]['lr'] = get_lr_anneal(args, iters)
    if hasattr(args, 'h_co_anneal') and args.h_co_anneal == "linear":
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

    # Add S2P Grad
    if args.s2p_freq > 0 and iters % args.s2p_freq == 0:
        fr_en_loss, en_de_loss = s2p_batch(s2p_fr_en_it, s2p_en_de_it, model)
        total_loss += args.s2p_co * (fr_en_loss + en_de_loss)

    total_loss.backward()
    if args.plot_grad:
        plot_grad(writer, model, iters)
    if args.grad_clip > 0:
        total_norm = nn.utils.clip_grad_norm_(params, args.grad_clip)
        if total_norm != total_norm or math.isnan(total_norm) or np.isnan(total_norm):
            print('NAN!!!!!!!!!!!!!!!!!!!!!!')
            exit()
    opt.step()

    if iters % args.eval_every == 0:
        args.logger.info("update {} : {}".format(iters, str(train_metrics)))
        if iters % args.eval_every == 0 and not args.debug:
            write_tb(writer, loss_names, [train_metrics.__getattr__(name) for name in loss_names],
                     iters, prefix="train/")
            write_tb(writer, monitor_names, [train_metrics.__getattr__(name) for name in monitor_names],
                     iters, prefix="train/")
            write_tb(writer, ['lr'], [opt.param_groups[0]['lr']], iters, prefix="train/")
            train_metrics.reset()


def itlearn_loop(args, model, train_it, dev_it, extra_input, loss_cos, loss_names, monitor_names):
    # Writer
    if not args.debug:
        decoding_path = Path(args.decoding_path + args.id_str)
        decoding_path.mkdir(parents=True, exist_ok=True)
        from tensorboardX import SummaryWriter
        writer = SummaryWriter( args.event_path + args.id_str)

    # Opt
    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == 'Adam':
        opt = torch.optim.Adam(params, betas=(0.9, 0.98), eps=1e-9, lr=args.lr)
    else:
        raise NotImplementedError

    train_metrics = Metrics('train_loss', *loss_names, *monitor_names, data_type="avg")
    best = Best(max, 'de_bleu', 'en_bleu', 'iters', model=model, opt=opt, path=args.model_path + args.id_str,
                gpu=args.gpu, debug=args.debug)

    # Prepare_init_student
    student = get_model(args)
    student.load_state_dict(model.state_dict())
    if torch.cuda.device_count() > 0 and args.gpu > -1:
        student.cuda(args.gpu)
    stu_fr_en_opt, stu_en_de_opt = None, None

    # S2P
    s2p_fr_en_it, s2p_en_de_it = None, None
    if hasattr(args, 's2p_freq') and args.s2p_freq > 0 and args.s2p_mode == "basic":
        args.logger.info('Perform S2P at every {} steps'.format(args.s2p_freq))
        s2p_fr_en_it, s2p_en_de_it = extra_input['s2p_its']['fr-en'], extra_input['s2p_its']['en-de']
        s2p_fr_en_it = iter(s2p_fr_en_it)
        s2p_en_de_it = iter(s2p_en_de_it)

    # Determine when to stop iterlearn
    max_itlearn_steps = args.max_itlearn_steps if args.max_itlearn_steps > 0 else args.max_training_steps
    for iters, train_batch in enumerate(train_it):
        if iters >= args.max_training_steps:
            args.logger.info('stopping training after {} training steps'.format(args.max_training_steps))
            break

        if not args.debug and iters in args.save_at:
            args.logger.info('save (back-up) checkpoints at iters={}'.format(iters))
            with torch.cuda.device(args.gpu):
                torch.save(model.state_dict(), '{}_iter={}.pt'.format(args.model_path + args.id_str, iters))
                torch.save([iters, opt.state_dict()],
                           '{}_iter={}.pt.states'.format(args.model_path + args.id_str, iters))

        if iters % args.eval_every == 0:
            dev_metrics = valid_model(model, dev_it, loss_names, monitor_names, extra_input)
            eval_metric, bleu_en, bleu_de = eval_model(args, model, dev_it, monitor_names, iters, extra_input)
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

        model.train()
        selfplay_step(args, extra_input, iters, loss_cos, loss_names, model, monitor_names, opt, params, train_batch,
                      train_metrics, writer, s2p_fr_en_it=s2p_fr_en_it,
                      s2p_en_de_it=s2p_en_de_it)

        if (iters + 1) % args.k1 == 0 and (iters + 1) < max_itlearn_steps:
            args.logger.info('start imitating at iters {}'.format(iters + 1))
            student.train()
            old_student_fr_en_stats = get_fr_en_imitate_stats(args, student, dev_it, monitor_names, extra_input)
            old_student_en_de_stats = get_en_de_imitate_stats(args, student, dev_it)
            model.eval()
            stu_fr_en_opt, stu_en_de_opt = get_student_opts(args, student, (stu_fr_en_opt, stu_en_de_opt))
            if not args.fr_en_reset:
                student.fr_en.load_state_dict(model.fr_en.state_dict())
            fr_en_statss = imitate_fr_en(args, student=student,
                                         teacher=model, train_it=train_it,
                                         dev_it=dev_it, monitor_names=monitor_names,
                                         extra_input=extra_input, opt=stu_fr_en_opt)

            if not args.en_de_finetune:
                en_de_statss = imitate_en_de(args, student=student,
                                             teacher=model, train_it=train_it, dev_it=dev_it,
                                             opt=stu_en_de_opt)
            else:
                en_de_statss = finetune_en_de(args, student=student,
                                              teacher=model, train_it=train_it, dev_it=dev_it,
                                              opt=stu_en_de_opt)

            # Report change of student and teacher
            teacher_fr_en_stats = get_fr_en_imitate_stats(args, model, dev_it, monitor_names, extra_input)
            student_fr_en_stats = get_fr_en_imitate_stats(args, student, dev_it, monitor_names, extra_input)
            teacher_en_de_stats = get_en_de_imitate_stats(args, model, dev_it)
            student_en_de_stats = get_en_de_imitate_stats(args, student, dev_it)
            df = DataFrame(columns=['teacher', 'stu', 'old_stu'])
            for name in student_fr_en_stats:
                df.loc[name, 'stu'] = student_fr_en_stats[name]
                df.loc[name, 'teacher'] = teacher_fr_en_stats[name]
                df.loc[name, 'old_stu'] = old_student_fr_en_stats[name]
            for name in student_en_de_stats:
                df.loc[name, 'stu'] = student_en_de_stats[name]
                df.loc[name, 'teacher'] = teacher_en_de_stats[name]
                df.loc[name, 'old_stu'] = old_student_en_de_stats[name]
            args.logger.info(str(df))

            if args.save_imitate_stats:
                print('Save imitation stats')
                if not os.path.exists(os.path.join(args.misc_path, args.id_str)):
                    os.makedirs(os.path.join(args.misc_path, args.id_str))
                fr_en_fig = plot_imitate_stats(teacher_fr_en_stats, fr_en_statss)
                fr_en_fig.savefig(os.path.join(args.misc_path, args.id_str, 'fr_en_{}_stats.png'.format(iters + 1)))
                del fr_en_fig
                en_de_fig = plot_imitate_stats(teacher_en_de_stats, en_de_statss)
                en_de_fig.savefig(os.path.join(args.misc_path, args.id_str, 'en_de_{}_stats.png'.format(iters + 1)))
                del en_de_fig

            # Update teacher with finalized student
            model.load_state_dict(student.state_dict())


def plot_imitate_stats(teacher_stats, imitate_statss):
    iterss = [res[0] for res in imitate_statss]
    statss = [res[1] for res in imitate_statss]
    fig, axs = plt.subplots(len(teacher_stats), figsize=(7, 7 * len(teacher_stats)))
    axs = axs.reshape(-1) if len(teacher_stats) > 1 else [axs]
    for name, ax in zip(teacher_stats, axs):
        student_vals = [stats[name] for stats in statss]
        teacher_val = teacher_stats[name]
        ax.plot(iterss, student_vals, label='student')
        ax.plot([iterss[0], iterss[-1]], [teacher_val, teacher_val], label='teacher')
        ax.set_title(name)
        ax.legend()
    return fig


def get_student_opts(args, student, student_opts=None):
    # Same opt
    if args.same_opt and student_opts[0] is not None and student_opts[1] is not None:
        args.logger.info('Reuse optimizer!')
        return student_opts

    # Create
    else:
        args.logger.info('Create new optimizer')
        fr_en_opt = torch.optim.Adam(student.fr_en.parameters(), betas=(0.9, 0.98),
                                     eps=1e-9, lr=args.fr_en_lr)
        en_de_opt = torch.optim.Adam(student.en_de.parameters(), betas=(0.9, 0.98),
                                     eps=1e-9, lr=args.en_de_lr)
        return fr_en_opt, en_de_opt


def s2p_batch(fr_en_it, en_de_it, agents):
    fr_en_batch = fr_en_it.__next__()
    en_de_batch = en_de_it.__next__()
    fr_en_loss = _get_nll(agents.fr_en,
                          fr_en_batch.src[0],
                          fr_en_batch.src[1],
                          fr_en_batch.trg[0])
    en_de_loss = _get_nll(agents.en_de,
                          en_de_batch.src[0],
                          en_de_batch.src[1],
                          en_de_batch.trg[0])
    return fr_en_loss, en_de_loss


def _get_nll(model, src, src_len, trg):
    logits, _ = model(src[:, 1:], src_len - 1, trg[:, :-1])
    loss = torch.nn.functional.cross_entropy(logits, trg[:, 1:].contiguous().view(-1),
                                             reduction='mean', ignore_index=0)
    return loss
