from pathlib import Path

import torch
import torch.nn.functional as F

from itlearn.utils.metrics import Metrics
from itlearn.utils.bleu import computeBLEU, print_bleu
from itlearn.utils.misc import cuda, sum_reward


def _supervise_loop(agent, dev_it, dataset='iwslt', pair='fr_en'):
    dev_metrics = Metrics('s2p_dev', ['nll'])
    with torch.no_grad():
        agent.eval()
        src_corpus, trg_corpus, hyp_corpus = [], [], []

        for j, dev_batch in enumerate(dev_it):
            if dataset == "iwslt" or dataset == 'iwslt_small':
                src, src_len = dev_batch.src
                trg, trg_len = dev_batch.trg
            elif dataset == "multi30k":
                src_lang, trg_lang = pair.split("_")
                src, src_len = dev_batch.__dict__[src_lang]
                trg, trg_len = dev_batch.__dict__[trg_lang]
            logits, _ = agent(src[:, 1:], src_len - 1, trg[:, :-1])
            nll = F.cross_entropy(logits, trg[:, 1:].contiguous().view(-1), reduction='mean',
                                  ignore_index=0)
            num_trg = (trg_len - 1).sum().item()
            dev_metrics.accumulate(num_trg, nll.item())
            hyp = agent.decode(src, src_len, 'argmax', 0)
            src_corpus.extend(agent.src.reverse(src))
            trg_corpus.extend(agent.trg.reverse(trg))
            hyp_corpus.extend(agent.trg.reverse(hyp))
        bleu = computeBLEU(hyp_corpus, trg_corpus, corpus=True)
    return dev_metrics, bleu


def eval_fr_en_stats(model, en_msg, en_msg_len, batch, en_lm=None, all_img=None, ranker=None,
                     use_gumbel_tokens=False):
    """ Evaluate this eng sentence with different metric. Can be used as reward """
    results = {}
    rewards = {}
    batch_size = en_msg.shape[0]
    # NOTE add <BOS> to beginning
    en_msg_ = torch.cat([cuda(torch.full((batch_size, 1), model.init_token)).long(), en_msg], dim=1)
    gumbel_tokens = None
    if use_gumbel_tokens:
        gumbel_tokens = model.fr_en.dec.gumbel_tokens
        init_tokens = torch.zeros([gumbel_tokens.shape[0], 1, gumbel_tokens.shape[2]])
        init_tokens = init_tokens.to(device=gumbel_tokens.device)
        init_tokens[:, :, model.init_token] = 1
        gumbel_tokens = torch.cat([init_tokens, gumbel_tokens], dim=1)

    if model.use_en_lm:  # monitor EN LM NLL
        if "wiki" in model.en_lm_dataset:
            if use_gumbel_tokens:
                raise NotImplementedError
            en_nll_lm = en_lm.get_nll(en_msg_)  # (batch_size, en_msg_len)
            if model.train_en_lm:
                en_nll_lm = sum_reward(en_nll_lm, en_msg_len + 1)  # (batch_size)
                rewards['lm'] = -1 * en_nll_lm.detach()
                # R = R + -1 * en_nll_lm.detach() * self.en_lm_nll_co # (batch_size)
            results.update({"en_nll_lm": en_nll_lm.mean()})

        elif model.en_lm_dataset in ["coco", "multi30k"]:
            if use_gumbel_tokens:
                en_lm.train()
                en_nll_lm = en_lm.get_loss_oh(gumbel_tokens, None)
                en_lm.eval()
            else:
                en_nll_lm = en_lm.get_loss(en_msg_, None)  # (batch_size, en_msg_len)
            if model.train_en_lm:
                en_nll_lm = sum_reward(en_nll_lm, en_msg_len + 1)  # (batch_size)
                rewards['lm'] = -1 * en_nll_lm.detach()
            results.update({"en_nll_lm": en_nll_lm.mean()})
        else:
            raise Exception()

    if model.use_ranker:  # NOTE Experiment 3 : Reward = NLL_DE + NLL_EN_LM + NLL_IMG_PRED
        if use_gumbel_tokens and model.train_ranker:
            raise NotImplementedError
        ranker.eval()
        img = cuda(all_img.index_select(dim=0, index=batch.idx.cpu()))  # (batch_size, D_img)

        if model.img_pred_loss == "nll":
            img_pred_loss = ranker.get_loss(en_msg_, img)  # (batch_size, en_msg_len)
            img_pred_loss = sum_reward(img_pred_loss, en_msg_len + 1)  # (batch_size)
        else:
            with torch.no_grad():
                img_pred_loss = ranker(en_msg, en_msg_len, img)["loss"]

        if model.train_ranker:
            rewards['img_pred'] = -1 * img_pred_loss.detach()
        results.update({"img_pred_loss_{}".format(model.img_pred_loss): img_pred_loss.mean()})

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

            if args.debug:
                break

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
            if model.debug:
                break
    return dev_metrics


def _base_train(args, model, iterators, extra_input, loop, loss_names, loss_cos):
    train_it, dev_it = iterators
    monitor_names = []
    if args.use_ranker:
        monitor_names.extend(["img_pred_loss_{}".format(args.img_pred_loss)])
        monitor_names.extend(["r1_acc"])
    if args.use_en_lm:
        monitor_names.append('en_nll_lm')
    loop(args=args, model=model, train_it=train_it, dev_it=dev_it,
         extra_input=extra_input, loss_cos=loss_cos, loss_names=loss_names,
         monitor_names=monitor_names)


def train_a2c_model(args, model, iterators, extra_input, loop):
    loss_names = ['ce_loss']
    loss_cos = {"ce_loss": args.ce_co}
    loss_names.extend(['pg_loss', 'b_loss', 'neg_Hs'])
    loss_cos.update({'pg_loss': args.pg_co, 'b_loss': args.b_co, 'neg_Hs': args.h_co})
    return _base_train(args, model, iterators, extra_input, loop, loss_names, loss_cos)


def train_gumbel_model(args, model, iterators, extra_input, loop):
    loss_names = ['ce_loss']
    loss_cos = {"ce_loss": args.ce_co}

    args.logger.info("Don't train entropy but observe it")
    loss_names.extend(['neg_Hs'])
    loss_cos.update({'neg_Hs': 0.})

    # Use LM reward
    if args.train_en_lm:
        args.logger.info('Train with LM reward {}'.format(args.en_lm_nll_co))
        loss_names.extend(['en_nll_lm'])
        loss_cos['en_nll_lm'] = args.en_lm_nll_co

        # Use entropy coef as well. If KL then h_co = en_lm_nll_co
        args.logger.info('Train entropy with {}'.format(args.h_co))
        loss_cos['neg_Hs'] = args.h_co

    if args.train_ranker:
        img_pred_loss_name = "img_pred_loss_{}".format(args.img_pred_loss)
        args.logger.info('Train with grounding reward')
        loss_names.extend([img_pred_loss_name])
        loss_cos[img_pred_loss_name] = args.img_pred_loss_co
    return _base_train(args, model, iterators, extra_input, loop, loss_names, loss_cos)
