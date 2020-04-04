import torch
import torch.nn as nn
from torch.nn import functional as F

from utils.misc import write_tb
from metrics import Metrics, Best
from utils.bleu import computeBLEU, print_bleu
from pathlib import Path
from os.path import join


def valid_model(args, model, dev_it, dev_metrics, decode_method, beam_width=5, test_set="valid"):
    with torch.no_grad():
        model.eval()
        src_corpus, trg_corpus, hyp_corpus = [], [], []

        for j, dev_batch in enumerate(dev_it):
            if args.dataset == "iwslt" or args.dataset == 'iwslt_small':
                src, src_len = dev_batch.src
                trg, trg_len = dev_batch.trg
            elif args.dataset == "multi30k":
                src_lang, trg_lang = args.pair.split("_")
                src, src_len = dev_batch.__dict__[src_lang]
                trg, trg_len = dev_batch.__dict__[trg_lang]
            logits, _ = model(src[:,1:], src_len-1, trg[:,:-1])
            nll = F.cross_entropy(logits, trg[:, 1:].contiguous().view(-1), reduction='mean',
                                  ignore_index=0)
            num_trg = (trg_len - 1).sum().item()
            dev_metrics.accumulate(num_trg, nll.item())
            hyp = model.decode(src, src_len, decode_method, beam_width)
            src_corpus.extend(args.src.reverse(src))
            trg_corpus.extend(args.trg.reverse(trg))
            hyp_corpus.extend(args.trg.reverse(hyp))

        bleu = computeBLEU(hyp_corpus, trg_corpus, corpus=True)
        args.logger.info(dev_metrics)
        args.logger.info("{} {} : {}".format(test_set, decode_method, print_bleu(bleu)))
        args.logger.info('model:' + args.prefix + args.hp_str)

        if not args.debug:
            src, trg, hyp = [Path(join(args.decoding_path, args.id_str, which))
                             for which in "src trg hyp".split()]
            src.write_text("\n".join(src_corpus), encoding="utf-8")
            trg.write_text("\n".join(trg_corpus), encoding="utf-8")
            hyp.write_text("\n".join(hyp_corpus), encoding="utf-8")
    return bleu


def train_model(args, model, iterators):
    (train_it, dev_it) = iterators

    if not args.debug:
        decoding_path = Path(join(args.decoding_path, args.id_str))
        decoding_path.mkdir(parents=True, exist_ok=True)
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(join(args.event_path, args.id_str))

    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == 'Adam':
        opt = torch.optim.Adam(params, betas=(0.9, 0.98), eps=1e-9, lr=args.lr)
    else:
        raise NotImplementedError

    extra_loss_names = []
    train_metrics = Metrics('train_loss', 'nll', *extra_loss_names, data_type="avg")
    dev_metrics = Metrics('dev_loss', 'nll', *extra_loss_names, data_type="avg")
    best = Best(max, 'dev_bleu', 'iters', model=model, opt=opt, path=join(args.model_path, args.id_str),
                gpu=args.gpu, debug=args.debug)

    for iters, train_batch in enumerate(train_it):
        if iters >= args.max_training_steps:
            args.logger.info('stopping training after {} training steps'.format(args.max_training_steps))
            break

        if not args.debug and iters % args.save_every == 0:
            args.logger.info('save (back-up) checkpoints at iters={}'.format(iters))
            with torch.cuda.device(args.gpu):
                torch.save(model.state_dict(), '{}_iter={}.pt'.format( args.model_path + args.id_str, iters))
                torch.save([iters, opt.state_dict()], '{}_iter={}.pt.states'.format( args.model_path + args.id_str, iters))

        if iters % args.eval_every == 0:
            dev_metrics.reset()
            dev_bleu = valid_model(args, model, dev_it, dev_metrics, args.decode_method)
            if not args.debug:
                write_tb(writer, ['nll'], [dev_metrics.nll], iters, prefix="dev/")
                write_tb(writer, ['bleu', *("p_1 p_2 p_3 p_4".split()) , 'bp', 'len_ref', 'len_hyp'], dev_bleu, iters, prefix="bleu/")
            best.accumulate(dev_bleu[0], iters)
            args.logger.info(best)

            """
            if args.early_stop and (iters - best.iters) // args.eval_every > args.patience:
                args.logger.info("Early stopping.")
                break
            """

        model.train()

        def get_lr_anneal(iters):
            lr_end = 1e-5
            return max( 0, (args.lr - lr_end) * (args.linear_anneal_steps - iters) / args.linear_anneal_steps ) + lr_end

        if args.lr_anneal == "linear":
            opt.param_groups[0]['lr'] = get_lr_anneal(iters)

        opt.zero_grad()

        batch_size = len(train_batch)
        if args.dataset == "iwslt" or args.dataset == 'iwslt_small':
            src, src_len = train_batch.src
            trg, trg_len = train_batch.trg
        elif args.dataset == "multi30k":
            src_lang, trg_lang = args.pair.split("_")
            src, src_len = train_batch.__dict__[src_lang]
            trg, trg_len = train_batch.__dict__[trg_lang]
        else:
            raise ValueError

        # NOTE encoder never receives <BOS> token 
        # because during communication, Agent A's decoder will never output <BOS>
        logits, _ = model(src[:,1:], src_len-1, trg[:,:-1])
        nll = F.cross_entropy(logits, trg[:, 1:].contiguous().view(-1), reduction='mean',
                              ignore_index=0)
        num_trg = (trg_len - 1).sum().item()
        train_metrics.accumulate(num_trg, nll.item())

        if args.grad_clip > 0:
            total_norm = nn.utils.clip_grad_norm_(params, args.grad_clip)
        nll.backward()
        opt.step()

        if iters % args.print_every == 0:
            args.logger.info("update {} : {}".format(iters, str(train_metrics)))
            if not args.debug:
                write_tb(writer, ['nll', 'lr'], [train_metrics.nll, opt.param_groups[0]['lr']], iters, prefix="train/")
            train_metrics.reset()

