import torch
import torch.nn as nn
from torch.nn import functional as F

from utils.misc import write_tb, plot_grad
from utils.metrics import Metrics, Best


def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def valid_model(args, model, dev_it, dev_metrics, iters):
    model.eval()
    hidden = model.init_hidden(args.batch_size)
    with torch.no_grad():
        for j, dev_batch in enumerate(dev_it):
            hidden = repackage_hidden(hidden)
            decoded, hidden = model(dev_batch.text, hidden, dev_batch.target)
            nll = F.cross_entropy( decoded, dev_batch.target.view(-1), ignore_index=0 )
            nll_cur = F.cross_entropy( decoded, dev_batch.text.view(-1), ignore_index=0 )
            dev_metrics.accumulate(len(dev_batch), nll.item(), nll_cur.item())

        args.logger.info(dev_metrics)

def train_model(args, model, iterators):
    (train_it, dev_it) = iterators

    if not args.debug:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter( args.event_path + args.id_str)

    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == 'Adam':
        opt = torch.optim.Adam(params, betas=(0.9, 0.98), eps=1e-9, lr=args.lr)
    elif args.optimizer == 'SGD':
        opt = torch.optim.SGD(params, lr=args.lr)
    else:
        raise NotImplementedError

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=args.anneal_by)

    loss_names, loss_cos = ["nll"], {"nll":1.0}
    monitor_names = []
    if args.rep_pen_co > 0.0:
        loss_names.append("nll_cur")
        loss_cos["nll_cur"] = -1 * args.rep_pen_co
    else:
        monitor_names.append("nll_cur")

    train_metrics = Metrics('train_loss', *loss_names, *monitor_names, data_type = "avg")
    dev_metrics = Metrics('dev_loss', *loss_names, *monitor_names, data_type = "avg")
    best = Best(min, 'loss', 'iters', model=model, opt=opt, path=args.model_path + args.id_str, \
                gpu=args.gpu, debug=args.debug)
    hidden = model.init_hidden(args.batch_size)

    for iters, train_batch in enumerate(train_it):

        if iters % args.eval_every == 0:
            dev_metrics.reset()
            valid_model(args, model, dev_it, dev_metrics, iters)
            if not args.debug:
                write_tb(writer, loss_names, [dev_metrics.__getattr__(name) for name in loss_names], \
                         iters, prefix="dev/")
                write_tb(writer, monitor_names, [dev_metrics.__getattr__(name) for name in monitor_names], \
                         iters, prefix="dev/")
            best.accumulate(dev_metrics.nll, iters)
            scheduler.step(dev_metrics.nll)

            args.logger.info('model:' + args.prefix + args.hp_str)
            args.logger.info(best)

            if args.early_stop and (iters - best.iters) // args.eval_every > args.patience:
                args.logger.info("Early stopping.")
                break

        model.train()

        opt.zero_grad()

        batch_size = len(train_batch)
        hidden = repackage_hidden(hidden)
        decoded, hidden = model(train_batch.text, hidden, train_batch.target)
        R = {}
        R["nll"] = F.cross_entropy( decoded, train_batch.target.view(-1), ignore_index=0 )
        R["nll_cur"] = F.cross_entropy( decoded, train_batch.text.view(-1), ignore_index=0 )

        total_loss = 0
        for loss_name in loss_names:
            total_loss += R[loss_name] * loss_cos[loss_name]

        train_metrics.accumulate(batch_size, *[R[name].item() for name in loss_names + monitor_names])

        total_loss.backward()
        if args.plot_grad:
            plot_grad(writer, model, iters)

        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(params, args.grad_clip)
        opt.step()

        if iters % args.eval_every == 0:
            args.logger.info("update {} : {}".format(iters, str(train_metrics)))

        if iters % args.eval_every == 0 and not args.debug:
            write_tb(writer, loss_names, [train_metrics.__getattr__(name) for name in loss_names], \
                     iters, prefix="train/")
            write_tb(writer, monitor_names, [train_metrics.__getattr__(name) for name in monitor_names], \
                     iters, prefix="train/")
            write_tb(writer, ['lr'], [opt.param_groups[0]['lr']], iters, prefix="train/")

            train_metrics.reset()

