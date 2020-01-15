from pathlib import Path

import torch

from metrics import Metrics
from misc.bleu import computeBLEU, print_bleu
from utils import cuda, sum_reward


def eval_fr_en_stats(args, en_msg, en_msg_len, batch, en_lm=None, all_img=None, ranker=None):
    """ Evaluate this eng sentence with different metric. Can be used as reward """
    results = {}
    rewards = {}
    batch_size = en_msg.shape[0]
    # NOTE add <BOS> to beginning
    en_msg_ = torch.cat([cuda(torch.full((batch_size, 1), args.init_token)).long(), en_msg], dim=1)
    if args.use_en_lm:  # monitor EN LM NLL
        if "wiki" in args.en_lm_dataset:
            en_nll_lm = en_lm.get_nll(en_msg_)  # (batch_size, en_msg_len)
            en_nll_lm = sum_reward(en_nll_lm, en_msg_len + 1)  # (batch_size)
            if args.train_en_lm:
                rewards['lm'] = -1 * en_nll_lm.detach()
                # R = R + -1 * en_nll_lm.detach() * self.en_lm_nll_co # (batch_size)
            results.update({"en_nll_lm": en_nll_lm.mean()})

        elif args.en_lm_dataset in ["coco", "multi30k"]:
            en_nll_lm = en_lm.get_loss(en_msg_, None)  # (batch_size, en_msg_len)
            en_nll_lm = sum_reward(en_nll_lm, en_msg_len + 1)  # (batch_size)
            if args.train_en_lm:
                rewards['lm'] = -1 * en_nll_lm.detach()
            results.update({"en_nll_lm": en_nll_lm.mean()})
        else:
            raise Exception()

    if args.use_ranker:  # NOTE Experiment 3 : Reward = NLL_DE + NLL_EN_LM + NLL_IMG_PRED
        img = cuda(all_img.index_select(dim=0, index=batch.idx.cpu()))  # (batch_size, D_img)

        if args.img_pred_loss == "nll":
            img_pred_loss = ranker.get_loss(en_msg_, img)  # (batch_size, en_msg_len)
            img_pred_loss = sum_reward(img_pred_loss, en_msg_len + 1)  # (batch_size)
        else:
            with torch.no_grad():
                img_pred_loss = ranker(en_msg, en_msg_len, img)["loss"]

        if args.train_ranker:
            rewards['img_pred'] = -1 * img_pred_loss.detach()
        results.update({"img_pred_loss_{}".format(args.img_pred_loss): img_pred_loss.mean()})
    return results, rewards


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
            results, _ = eval_fr_en_stats(model, en_msg, en_msg_len, dev_batch,
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


def _base_train(args, model, iterators, extra_input, loop, is_a2c=False):
    (train_it, dev_it) = iterators
    monitor_names = []
    loss_names = ['ce_loss']
    loss_cos = {"ce_loss": args.ce_co}

    # Extra loss for a2c
    if is_a2c:
        loss_names.extend(['pg_loss', 'b_loss', 'neg_Hs'])
        loss_cos.update({'pg_loss': args.pg_co, 'b_loss': args.b_co, 'neg_Hs': args.h_co})

    # Monitor the entropy of gumbel but don't train
    else:
        args.logger.info("Don't train entropy but observe it")
        loss_names.extend(['neg_Hs'])
        loss_cos.update({'neg_Hs': 0.})

    if args.use_ranker:
        monitor_names.extend(["img_pred_loss_{}".format(args.img_pred_loss)])
    if args.use_en_lm:
        monitor_names.append('en_nll_lm')
    loop(args=args, model=model, train_it=train_it, dev_it=dev_it,
         extra_input=extra_input, loss_cos=loss_cos, loss_names=loss_names,
         monitor_names=monitor_names)


def train_a2c_model(args, model, iterators, extra_input, loop):
    return _base_train(args, model, iterators, extra_input, loop, is_a2c=True)


def train_gumbel_model(args, model, iterators, extra_input, loop):
    return _base_train(args, model, iterators, extra_input, loop, is_a2c=False)
