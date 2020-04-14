"""
Evaluate the ease of teaching accross the training trajectory
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
from tensorboardX import SummaryWriter
from shutil import rmtree

from utils.metrics import Metrics
from utils.misc import write_tb
from utils.bleu import computeBLEU, print_bleu
from utils.hyperparams import Params
from data import get_multi30k_iters, batch_size_fn
from torchtext.data import Dataset, Example, BucketIterator, interleave_keys
from finetune.agents import AgentsGumbel
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--exp_dir', required=True)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--nb_runs', default=5, type=int)
    parser.add_argument('--learning_steps', default=10000, type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--normalized', action='store_true')
    parser.add_argument('--remove_dots', action='store_true')
    parser.add_argument('--human', action='store_true')
    parsed_args = parser.parse_args().__dict__

    print('###### CONFIG #########')
    for key, val in parsed_args.items():
        print('{}: {}'.format(key, val))
    print('#######################')

    # Try to load json
    json_dir = os.path.join(parsed_args['exp_dir'], 'param', parsed_args['experiment'])
    fnames = os.listdir(json_dir)
    assert len(fnames) == 1, "More than one config file in the experiment!"
    print('Find json_config', fnames[0])
    params = Params(os.path.join(json_dir, fnames[0]))

    # Create output dir
    if 'outdir' in parsed_args:
        print('storing result in {}'.format(parsed_args['outdir']))
        if os.path.exists(parsed_args['outdir']):
            rmtree(parsed_args['outdir'])
        os.makedirs(parsed_args['outdir'])
    else:
        raise ValueError('You must pass in --outdir')

    # Update some of them with command line
    params.update(parsed_args)
    return params


def build_fr_en_it(multi30_it, train_repeat, batch_size, device, teacher_model=None, remove_dots=False):
    """ Build an iterator from multi30k_data
    with a teacher model for france to english. Use original Eng if teacher is None """
    fr, en = multi30_it.dataset.fields['fr'], multi30_it.dataset.fields['en']
    if teacher_model is not None:
        teacher_model.eval()
        fr_corpus, en_corpus = [], []
        for batch in multi30_it:
            en_msg, _ = teacher_model.fr_en_speak(batch)
            fr_corpus.extend(fr.reverse(batch.fr[0], unbpe=True))
            en_corpus.extend(en.reverse(en_msg, unbpe=True, remove_dots=remove_dots))
    else:
        en_corpus, fr_corpus = [], []
        for batch in multi30_it:
            fr_corpus.extend(fr.reverse(batch.fr[0], unbpe=True))
            en_corpus.extend(en.reverse(batch.en[0], unbpe=True, remove_dots=remove_dots))
    fields = [('src', fr), ('trg', en)]
    exs = [Example.fromlist(data=[fr_sent, en_sent], fields=fields)
           for fr_sent, en_sent in zip(fr_corpus, en_corpus)]
    dataset = Dataset(examples=exs, fields=fields)
    it = BucketIterator(dataset, batch_size, device=device,
                        batch_size_fn=batch_size_fn,
                        repeat=train_repeat, shuffle=True,
                        sort=False, sort_within_batch=True,
                        sort_key=lambda ex: interleave_keys(len(ex.src), len(ex.trg)))
    return it


def valid_model(model, dev_it, dev_metrics):
    fields = dev_it.dataset.fields
    with torch.no_grad():
        model.eval()
        trg_corpus, hyp_corpus = [], []

        for j, dev_batch in enumerate(dev_it):
            src, src_len = dev_batch.src
            trg, trg_len = dev_batch.trg
            logits, _ = model(src[:, 1:], src_len-1, trg[:, :-1])
            nll = F.cross_entropy(logits, trg[:, 1:].contiguous().view(-1), reduction='mean',
                                  ignore_index=0)
            num_trg = (trg_len - 1).sum().item()
            dev_metrics.accumulate(num_trg, nll.item())
            hyp = model.decode(src, src_len, "greedy", 0)
            trg_corpus.extend(fields['trg'].reverse(trg, True))
            hyp_corpus.extend(fields['trg'].reverse(hyp, True))

        bleu = computeBLEU(hyp_corpus, trg_corpus, corpus=True)
        print((dev_metrics))
        print(print_bleu(bleu))
    return bleu


def train_model(model, train_it, dev_it, outdir, max_training_steps):
    """ supervise learning on the dataset """
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, betas=(0.9, 0.98), eps=1e-9, lr=1e-4)
    train_metrics = Metrics('train_loss', 'nll', data_type="avg")
    dev_metrics = Metrics('dev_loss', 'nll', data_type="avg")
    writer = SummaryWriter(os.path.join(outdir, 'logs'))

    for iters, train_batch in enumerate(train_it):
        if iters >= max_training_steps:
            print('stopping training after {} training steps'.format(max_training_steps))
            break

        if iters % 1000 == 0:
            dev_metrics.reset()
            dev_bleu = valid_model(model, dev_it, dev_metrics)
            write_tb(writer, ['nll'], [dev_metrics.nll], iters, prefix="dev/")
            write_tb(writer, ['bleu', *("p_1 p_2 p_3 p_4".split()), 'bp', 'len_ref', 'len_hyp'],
                     dev_bleu, iters, prefix="bleu/")

        model.train()
        src, src_len = train_batch.src
        trg, trg_len = train_batch.trg
        logits, _ = model(src[:,1:], src_len-1, trg[:,:-1])
        nll = F.cross_entropy(logits, trg[:, 1:].contiguous().view(-1), reduction='mean',
                              ignore_index=0)
        num_trg = (trg_len - 1).sum().item()
        train_metrics.accumulate(num_trg, nll.item())

        opt.zero_grad()
        nll.backward()
        total_norm = nn.utils.clip_grad_norm_(params, 0.2)
        opt.step()

        if iters % 200 == 0:
            print("update {} : {}".format(iters, str(train_metrics)))
            write_tb(writer, ['nll', 'lr'], [train_metrics.nll, opt.param_groups[0]['lr']], iters, prefix="train/")
            train_metrics.reset()

    # Final evaluation
    dev_metrics.reset()
    dev_bleu = valid_model(model, dev_it, dev_metrics)
    write_tb(writer, ['nll'], [dev_metrics.nll], iters, prefix="dev/")
    write_tb(writer, ['bleu', *("p_1 p_2 p_3 p_4".split()), 'bp', 'len_ref', 'len_hyp'],
             dev_bleu, iters, prefix="bleu/")
    writer.flush()

    # Return stats
    if not args.normalized:
        # Unnormalized
        stats = {'dev/{}'.format(key): dev_metrics.metrics[key] for key in dev_metrics.metrics}
        stats.update({'train/{}'.format(key): train_metrics.metrics[key] for key in train_metrics.metrics})
    else:
        # Normalized
        stats = {'dev/{}'.format(key): dev_metrics.__getattr__(key) for key in dev_metrics.metrics}
        stats.update({'train/{}'.format(key): train_metrics.__getattr__(key) for key in train_metrics.metrics})
    stats['dev/bleu'] = dev_bleu[0]
    return stats


def main():
    # Read a list of checkpoint
    ckpt_dir = os.path.join(args.exp_dir, 'model', args.experiment)
    ckpts = [m for m in os.listdir(ckpt_dir) if 'iter=' in m and '.states' not in m]
    ckpt_with_steps = {int(ckpt.split('.pt')[0].split('_iter=')[-1]): ckpt
                       for ckpt in ckpts}
    if args.debug:
        print('Debug mode on')
        ckpt_with_steps = {step: ckpt_with_steps[step] for step in sorted(ckpt_with_steps.keys())[:5]}
    else:
        print('Debug mode off')
    print('Found {} checkpints'.format(len(ckpt_with_steps)))

    # Data
    device = "cuda:{}".format(args.gpu) if args.gpu > -1 else "cpu"
    bpe_path = os.path.join(args.data_dir, 'bpe')
    en_vocab, de_vocab, fr_vocab = "vocab.en.pth", "vocab.de.pth", "vocab.fr.pth"
    de, en, fr, orig_train_it, orig_dev_it = get_multi30k_iters(bpe_path=bpe_path, de_vocab=de_vocab, device=device,
                                                                en_vocab=en_vocab, fr_vocab=fr_vocab,
                                                                train_repeat=False,
                                                                load_dataset=False,
                                                                save_dataset=False,
                                                                batch_size=512)
    args.__dict__.update({'FR': fr, "DE": de, "EN": en})
    args.__dict__.update({'pad_token': 0,
                          'unk_token': 1,
                          'init_token': 2,
                          'eos_token': 3})

    # Models
    teacher = AgentsGumbel(args)
    if args.gpu > -1 and torch.cuda.device_count() > 0:
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = lambda storage, loc: storage
    teacher.cuda(args.gpu)

    statss = []
    for iter_step in sorted(ckpt_with_steps.keys()):
        ckpt = ckpt_with_steps[iter_step]
        teacher.load_state_dict(torch.load(os.path.join(args.exp_dir, 'model', args.experiment, ckpt),
                                           map_location=map_location))
        print('################################################')
        print('Load teacher from iter={}'.format(iter_step))

        # Generate New iterator
        remove_dots = hasattr(args, 'remove_dots') and args.remove_dots
        dev_it = build_fr_en_it(orig_dev_it, teacher_model=teacher, train_repeat=False,
                                device=device, batch_size=512,
                                remove_dots=remove_dots)
        print('Build Dev dataset done')

        if args.debug:
            train_it = build_fr_en_it(orig_dev_it, teacher_model=teacher, train_repeat=True,
                                      device=device, batch_size=512, remove_dots=remove_dots)
        else:
            train_it = build_fr_en_it(orig_train_it, teacher_model=teacher, train_repeat=True,
                                      device=device, batch_size=512, remove_dots=remove_dots)
        print('Build Train dataset done')

        runs_stats = []
        for run in range(args.nb_runs):
            # Start learning
            student = AgentsGumbel(args)
            student.cuda(args.gpu)
            print('Initial model done')

            stats = train_model(student.fr_en, train_it, dev_it,
                                outdir=os.path.join(args.outdir, 'iters_{}_run_{}'.format(iter_step, run)),
                                max_training_steps=100 if args.debug else args.learning_steps)
            runs_stats.append(stats)
        statss.append(runs_stats)

    # Save to pickle
    import pickle
    with open(os.path.join(args.outdir, 'data.pkl'), 'wb') as f:
        pickle.dump({'steps': sorted(ckpt_with_steps.keys()), 'statss': statss}, f)

    import matplotlib.pyplot as plt
    import numpy as np
    print('Start plotting...')
    NB_COL = 2
    NB_ROW = int((len(statss[0]) + 1)/2)
    fig, axs = plt.subplots(NB_ROW, 2, figsize=(8*NB_ROW, 10*NB_COL))
    for key, ax in zip(statss[0][0], axs.reshape(-1)):
        steps = sorted(ckpt_with_steps.keys())
        values = [[stats[key] for stats in run_stats] for run_stats in statss]
        means = np.mean(values, -1)
        stds = np.std(values, -1)
        ax.plot(steps, means)
        ax.fill_between(steps, means - stds, means + stds,
                        alpha=0.2)
        ax.set_xlabel('steps', fontsize=15)
        ax.set_title(key)

    fig.savefig(os.path.join(args.outdir, 'plots.png'))


def human_eot():
    # Data
    device = "cuda:{}".format(args.gpu) if args.gpu > -1 else "cpu"
    bpe_path = os.path.join(args.data_dir, 'bpe')
    en_vocab, de_vocab, fr_vocab = "vocab.en.pth", "vocab.de.pth", "vocab.fr.pth"
    de, en, fr, orig_train_it, orig_dev_it = get_multi30k_iters(bpe_path=bpe_path, de_vocab=de_vocab, device=device,
                                                                en_vocab=en_vocab, fr_vocab=fr_vocab,
                                                                train_repeat=False,
                                                                load_dataset=False,
                                                                save_dataset=False,
                                                                batch_size=512)
    args.__dict__.update({'FR': fr, "DE": de, "EN": en})
    args.__dict__.update({'pad_token': 0,
                          'unk_token': 1,
                          'init_token': 2,
                          'eos_token': 3})

    print('################################################')

    # Generate New iterator
    dev_it = build_fr_en_it(orig_dev_it, teacher_model=None, train_repeat=False,
                            device=device, batch_size=512,
                            remove_dots=args.remove_dots)
    print('Build Dev dataset done')

    if args.debug:
        train_it = build_fr_en_it(orig_dev_it, teacher_model=None, train_repeat=True,
                                  device=device, batch_size=512, remove_dots=args.remove_dots)
    else:
        train_it = build_fr_en_it(orig_train_it, teacher_model=None, train_repeat=True,
                                  device=device, batch_size=512, remove_dots=args.remove_dots)
    print('Build Train dataset done')

    runs_stats = []
    for run in range(args.nb_runs):
        # Start learning
        student = AgentsGumbel(args)
        student.cuda(args.gpu)
        print('Initial model done')

        stats = train_model(student.fr_en, train_it, dev_it,
                            outdir=os.path.join(args.outdir, 'human_run_{}'.format(run)),
                            max_training_steps=100 if args.debug else args.nb_runs)
        runs_stats.append(stats)

    # Save to pickle
    import pickle
    with open(os.path.join(args.outdir, 'data.pkl'), 'wb') as f:
        pickle.dump({'human_stats': runs_stats}, f)


if __name__ == '__main__':
    args = get_args()
    if 'human' in args.__dict__ and args.human:
        print('Run Human EOT result')
        human_eot()
    else:
        print('Run checkpoint EOT result')
        main()


