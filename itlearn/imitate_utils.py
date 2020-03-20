import torch
from torch.nn import functional as F
import random

from agents_utils import eval_fr_en_stats
from metrics import Metrics
from misc.bleu import computeBLEU
from utils import cuda

__all__ = ['imitate_fr_en', 'imitate_en_de', 'finetune_en_de',
           'get_fr_en_imitate_stats', 'get_en_de_imitate_stats']


def _fr_en_s2p_iwslt(batch, model):
    model.train()
    src, src_len = batch.src
    trg = batch.trg[0]
    logits, _ = model.fr_en(src[:, 1:], src_len - 1, trg[:, :-1])
    nll = torch.nn.functional.cross_entropy(logits, trg[:, 1:].contiguous().view(-1),
                                            reduction='mean', ignore_index=0)
    return nll


def _fr_en_s2p_multi30k(batch, model):
    model.train()
    (fr, fr_len) = batch.fr
    (en, _) = batch.en
    logits, _ = model.fr_en(batch.fr[:, 1:], fr_len - 1, en[:, :1-1])
    nll = torch.nn.functional.cross_entropy(logits, trg[:, 1:].contiguous().view(-1),
                                            reduction='mean', ignore_index=0)
    return nll


def _fr_en_imitate_step(args, batch, student, teacher):
    with torch.no_grad():
        teacher.eval()
        if args.send_method == 'argmax':
            en_msg, en_msg_len = teacher.fr_en_speak(batch, is_training=False)
        elif args.send_method == 'gumbel':
            en_msg, en_msg_len = teacher.fr_en_speak(batch, is_training=True)
        else:
            raise ValueError
        en_msg, en_msg_len = _make_sure_message_valid(en_msg, en_msg_len, teacher.init_token)
    student.train()
    nll = _get_imitate_loss(args.fr_en_temp, student.fr_en, teacher.fr_en, batch.fr[0], batch.fr[1], en_msg)
    return nll


def imitate_fr_en(args, student, teacher, train_it, dev_it, monitor_names, extra_input, opt):
    """ Imitate speake """
    args.logger.info('Fr En: Imitate {} S2P {}'.format(1 - args.fr_en_s2p_ratio,
                                                       args.fr_en_s2p_ratio))
    imitate_statss = []
    eval_freq = max(int(args.fr_en_k2 / 50), 5)
    iters = 0
    s2p_fr_en_it = iter(extra_input['s2p_its']['fr-en'])
    train_it = iter(train_it)
    while True:
        if iters >= args.fr_en_k2:
            args.logger.info('student fr en stop learning after {} training steps'.format(args.fr_en_k2))
            break

        if args.save_imitate_stats and iters % eval_freq == 0:
            args.logger.info('Record imitate stats at {}'.format(iters))
            student.eval()
            stats = get_fr_en_imitate_stats(args, student, dev_it, monitor_names, extra_input)
            imitate_statss.append((iters, stats))

        # This mode of S2P will use probability to do separate update
        if args.fr_en_s2p_mode == 'prob':
            if random.random() < args.fr_en_s2p_ratio:
                # Use labeled data
                if args.fr_en_s2p_dataset == 'iwslt':
                    batch = s2p_fr_en_it.__next__()
                    loss = _fr_en_s2p_iwslt(batch, student)
                elif args.fr_en_s2p_dataset == 'multi30k':
                    batch = train_it.__next__()
                    loss = _fr_en_s2p_multi30k(batch, student)
                else:
                    raise ValueError
            else:
                # Imitate
                batch = train_it.__next__()
                loss = _fr_en_imitate_step(args, batch, student, teacher)

        elif args.fr_en_s2p_mode == 'interp':
            if args.fr_en_s2p_dataset == 'iwslt':
                batch = s2p_fr_en_it.__next__()
                s2p_loss = _fr_en_s2p_iwslt(batch, student)
            elif args.fr_en_s2p_dataset == 'multi30k':
                batch = train_it.__next__()
                s2p_loss = _fr_en_s2p_multi30k(batch, student)
            else:
                raise ValueError

            batch = train_it.__next__()
            sil_loss = _fr_en_imitate_step(args, batch, student, teacher)
            loss = args.fr_en_s2p_ratio * s2p_loss + (1 - args.fr_en_s2p_ratio) * sil_loss
        else:
            raise ValueError

        # opt step
        opt.zero_grad()
        loss.backward()
        opt.step()
        iters += 1
    return imitate_statss


def get_fr_en_imitate_stats(args, model, dev_it, monitor_names, extra_input):
    """ En BLUE, LM score and img prediction """
    model.eval()
    eval_metrics = Metrics('dev_loss', *monitor_names, data_type="avg")
    eval_metrics.reset()
    with torch.no_grad():
        unbpe = True
        en_corpus = []
        en_hyp = []

        for j, dev_batch in enumerate(dev_it):
            en_corpus.extend(args.EN.reverse(dev_batch.en[0], unbpe=unbpe))
            en_msg, en_msg_len = model.fr_en_speak(dev_batch, is_training=False)
            en_hyp.extend(args.EN.reverse(en_msg, unbpe=unbpe))
            results, _ = eval_fr_en_stats(model, en_msg, en_msg_len, dev_batch,
                                          en_lm=extra_input["en_lm"],
                                          all_img=extra_input["img"]['multi30k'][1],
                                          ranker=extra_input["ranker"])
            if len(monitor_names) > 0:
                eval_metrics.accumulate(len(dev_batch), *[results[k].item() for k in monitor_names])

        bleu_en = computeBLEU(en_hyp, en_corpus, corpus=True)
        stats = eval_metrics.__dict__['metrics']
        stats['bleu_en'] = bleu_en[0]
        return stats


def finetune_en_de(args, student, teacher, train_it, dev_it, opt):
    """ Perform finetuning """
    args.logger.info('Finetune en de')
    imitate_statss = []
    eval_freq = max(int(args.en_de_k2 / 50), 5)
    for iters, batch in enumerate(train_it):
        if iters >= args.en_de_k2:
            args.logger.info('student en de stop finetuning after {} steps'.format(args.en_de_k2))
            break

        if args.save_imitate_stats and iters % eval_freq == 0:
            args.logger.info('Record imitate stats at {}'.format(iters))
            student.eval()
            stats = get_en_de_imitate_stats(args, student, dev_it)
            imitate_statss.append((iters, stats))

        # Teacher generate message
        with torch.no_grad():
            teacher.eval()
            if args.send_method == 'argmax':
                en_msg, en_msg_len = teacher.fr_en_speak(batch, is_training=False)
            elif args.send_method == 'gumbel':
                en_msg, en_msg_len = teacher.fr_en_speak(batch, is_training=True)
            else:
                raise ValueError
            en_msg, en_msg_len = _make_sure_message_valid(en_msg, en_msg_len, teacher.init_token)

        student.train()
        nll = _get_imitate_loss(0, student.en_de, teacher.en_de, en_msg, en_msg_len, batch.de[0])
        opt.zero_grad()
        nll.backward()
        opt.step()
    return imitate_statss


def imitate_en_de(args, student, teacher, train_it, dev_it, opt):
    args.logger.info('Imitate fr en!')
    imitate_statss = []
    eval_freq = max(int(args.en_de_k2 / 50), 5)
    for iters, batch in enumerate(train_it):
        if iters >= args.en_de_k2:
            args.logger.info('student en de stop learning after {} steps'.format(args.en_de_k2))
            break

        if args.save_imitate_stats and iters % eval_freq == 0:
            args.logger.info('Record imitate stats at {}'.format(iters))
            student.eval()
            stats = get_en_de_imitate_stats(args, student, dev_it)
            imitate_statss.append((iters, stats))

        # Teacher generate message
        with torch.no_grad():
            teacher.eval()
            if args.send_method == 'argmax':
                en_msg, en_msg_len = teacher.fr_en_speak(batch, is_training=False)
            elif args.send_method == 'gumbel':
                en_msg, en_msg_len = teacher.fr_en_speak(batch, is_training=True)
            else:
                raise ValueError
            en_msg, en_msg_len = _make_sure_message_valid(en_msg, en_msg_len, teacher.init_token)
            en_hid = teacher.en_de.enc(en_msg, en_msg_len)
            de_send_results = teacher.en_de.dec.send(en_hid, en_msg_len, batch.de[1] - 1, 'argmax')
            de_msg, de_msg_len = [de_send_results[key] for key in ["msg", "new_seq_lens"]]
            de_msg, de_msg_len = _make_sure_message_valid(de_msg, de_msg_len, teacher.init_token)

        student.train()
        # Imitate hard/soft labels
        if not args.en_de_finetune:
            nll = _get_imitate_loss(args.en_de_temp, student.en_de, teacher.en_de, en_msg, en_msg_len, de_msg)

        # Teacher forcing correct labels
        else:
            nll = _get_imitate_loss(0, student.en_de, teacher.en_de, en_msg, en_msg_len, batch.de[0])
        opt.zero_grad()
        nll.backward()
        opt.step()
    return imitate_statss


def get_en_de_imitate_stats(args, model, dev_it):
    with torch.no_grad():
        unbpe = True
        model.eval()
        de_corpus = []
        de_hyp = []
        for j, dev_batch in enumerate(dev_it):
            de_corpus.extend(args.DE.reverse(dev_batch.de[0], unbpe=unbpe))
            _, de_msg, _, de_msg_len = model.decode(dev_batch)
            de_hyp.extend(args.DE.reverse(de_msg, unbpe=unbpe))
        bleu_de = computeBLEU(de_hyp, de_corpus, corpus=True)
    return {'bleu_de': bleu_de[0]}


def _make_sure_message_valid(msg, msg_len, init_token):
    # Add BOS
    msg = torch.cat([cuda(torch.full((msg.shape[0], 1), init_token)).long(), msg],
                    dim=1)
    msg_len += 1

    # Make sure padding are all zeros
    #inv_mask = xlen_to_inv_mask(msg_len, seq_len=msg.shape[1])
    #msg.masked_fill_(mask=inv_mask.bool(), value=0)
    return msg, msg_len


def _get_imitate_loss(distill_temp, student_model, teacher_model, src, src_len, trg):
    """ Single model get NLL loss"""
    # NOTE encoder never receives <BOS> token
    # because during communication, Agent A's decoder will never output <BOS>
    student_logits, _ = student_model(src[:, 1:], src_len - 1, trg[:, :-1])

    # Loss is just nll
    if distill_temp == 0:
        loss = F.cross_entropy(student_logits, trg[:, 1:].contiguous().view(-1), reduction='mean',
                               ignore_index=0)

    # min KL(teacher | student) = p_teacher (log p - log q)
    else:
        with torch.no_grad():
            teacher_model.eval()
            teacher_logits, _ = teacher_model(src[:, 1:], src_len - 1, trg[:, :-1])
        teacher_probs = torch.nn.functional.softmax(teacher_logits, dim=-1)
        student_logprobs = torch.nn.functional.log_softmax(student_logits, dim=-1)
        kl = teacher_probs * (torch.log(teacher_probs) - student_logprobs)
        kl = kl.sum(-1) # [bsz, trg_len - 1]
        inv_mask = (trg[:, 1:] == 0).view(-1)
        kl.masked_fill_(inv_mask, 0)
        loss = kl.sum() / inv_mask.logical_not().sum()
    return loss
