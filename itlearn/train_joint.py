from torch import nn as nn
import torch
from pathlib import Path
from metrics import Metrics, Best
from utils import write_tb, plot_grad
from agents_utils import eval_model, valid_model
import math
import numpy as np


def joint_loop(args, model, train_it, dev_it, extra_input, loss_cos, loss_names, monitor_names):
    # Prepare writer
    if not args.debug:
        decoding_path = Path(args.decoding_path + args.id_str)
        decoding_path.mkdir(parents=True, exist_ok=True)
        from tensorboardX import SummaryWriter
        writer = SummaryWriter( args.event_path + args.id_str)

    # Prepare opt
    if args.fix_fr2en:
        args.logger.info('Fix Fr En')
        params = [p for p in model.en_de.parameters() if p.requires_grad]
    else:
        params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == 'Adam':
        opt = torch.optim.Adam(params, betas=(0.9, 0.98), eps=1e-9, lr=args.lr)
    else:
        raise NotImplementedError

    train_metrics = Metrics('train_loss', *loss_names, *monitor_names, data_type="avg")
    best = Best(max, 'de_bleu', 'en_bleu', 'iters', model=model, opt=opt, path=args.model_path + args.id_str,
                gpu=args.gpu, debug=args.debug)

    fr_en_it, en_de_it = None, None
    if hasattr(args, 's2p_freq') and args.s2p_freq > 0:
        args.logger.info('Perform S2P at every {} steps'.format(args.s2p_freq))
        fr_en_it, en_de_it = extra_input['s2p_its']['fr-en'], extra_input['s2p_its']['en-de']
        fr_en_it = iter(fr_en_it)
        en_de_it = iter(en_de_it)

    for iters, train_batch in enumerate(train_it):
        if iters >= args.max_training_steps:
            args.logger.info('stopping training after {} training steps'.format(args.max_training_steps))
            break

        if not args.debug and iters % args.save_every == 0:
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

        def get_lr_anneal(iters):
            lr_end = args.lr_min
            return max(0, (args.lr - lr_end) * (args.linear_anneal_steps - iters) / args.linear_anneal_steps) + lr_end

        def get_h_co_anneal(iters):
            h_co_end = args.h_co_min
            return max(0, (args.h_co - h_co_end) * (args.h_co_anneal_steps - iters) / args.h_co_anneal_steps) + h_co_end

        if hasattr(args, 'lr_anneal') and args.lr_anneal == "linear":
            opt.param_groups[0]['lr'] = get_lr_anneal(iters)

        if hasattr(args, 'h_co_anneal') and args.h_co_anneal == "linear":
            loss_cos['neg_Hs'] = get_h_co_anneal(iters)

        opt.zero_grad()

        batch_size = len(train_batch)
        R = model(train_batch, en_lm=extra_input["en_lm"], all_img=extra_input["img"]['multi30k'][0],
                  ranker=extra_input["ranker"])
        losses = [R[key] for key in loss_names]

        total_loss = 0
        for loss_name, loss in zip(loss_names, losses):
            assert loss.grad_fn is not None
            total_loss += loss * loss_cos[loss_name]

        train_metrics.accumulate(batch_size, *[loss.item() for loss in losses], *[R[k].item() for k in monitor_names])
        # Add S2P Grad
        if args.s2p_freq > 0 and iters % args.s2p_freq == 0:
            fr_en_loss, en_de_loss = s2p_batch(fr_en_it, en_de_it, model)
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
