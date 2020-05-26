"""
This script takes a tf board folder structure like
exp
|__exp1
|    |__ run1
|    |__ run2
|    |__ run1
|    |__ run1
|    |__ run3
|    |__ ...
|    |__ runN
|
|__exp2
     |__ run1
     |__ run2
     |__ run3
     |__ ...
     |__ runN

And generate plots with shaded area for all
"""
import argparse
import os
import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math

# default tags and names
from itlearn.utils.viz import combine_series, Series, parse_tb_event_files

TAGS = ['eval_bleu_de', 'eval_bleu_en', 'eval_en_nll_lm', 'eval_r1_acc', 'eval_nll_real', 'dev_neg_Hs']
NAMES = ['BLEU_De', 'BLEU_En', 'NLL', 'R1', 'Real NLL', 'Neg Entropy']


def plot_each_tag(all_data, ax, exp_names, max_steps, tag, font_size, use_median, tags, names):
    data = {}
    min_steps = math.inf
    for exp_name in exp_names:
        steps, means, stds = combine_series([run_data[tag] for run_data in all_data[exp_name]],
                                            use_median=use_median)
        data[exp_name] = (steps, means, stds)
        if steps[-1] < min_steps:
            min_steps = steps[-1]
    # Use Plot until max_steps + 10k
    for exp_name in exp_names:
        steps, means, stds = data[exp_name]
        plot_steps = min(max_steps[exp_name] + 1000, steps[-1])
        new_steps, new_means, new_stds = [], [], []
        for step, mean, std in zip(steps, means, stds):
            new_steps.append(step)
            new_means.append(mean)
            new_stds.append(std)
            # if step <= plot_steps:
            #    new_steps.append(step)
            #    new_means.append(mean)
            #    new_stds.append(std)
            # else:
            #    break
        new_means = np.array(new_means)
        new_stds = np.array(new_stds)
        line, = ax.plot(new_steps, new_means)
        line.set_label(exp_name)
        ax.fill_between(new_steps, new_means - new_stds, new_means + new_stds,
                        alpha=0.2)
    ax.set_xlabel('steps', fontsize=font_size)
    ax.set_title(names[tags.index(tag)])
    ax.legend(fontsize=20)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_dir')
    parser.add_argument('output_dir')
    parser.add_argument('-same_canvas', action='store_true')
    parser.add_argument('-use_median', action='store_true')
    parser.add_argument('-tags', nargs='*')
    parser.add_argument('-names', nargs='*')
    args = parser.parse_args()
    tags = args.tags if args.tags is not None and len(args.tags) > 0 else TAGS
    names = args.names if args.names is not None and len(args.names) > 0 else NAMES
    assert len(tags) == len(names)
    for name, tag in zip(names, tags):
        print('Plot {} ({})'.format(name, tag))

    exp_names = [exp_name for exp_name in sorted(os.listdir(args.exp_dir)) if
                 os.path.isdir(os.path.join(args.exp_dir, exp_name))]
    print('We have {} experiments'.format(len(exp_names)))

    all_data = {}
    all_tags = []
    for exp_name in exp_names:
        all_data[exp_name] = []
        runs = os.listdir(os.path.join(args.exp_dir, exp_name))
        print('We have {} runs for {}'.format(len(runs), exp_name))
        for run in runs:
            run_data = parse_tb_event_files(os.path.join(args.exp_dir, exp_name, run), tags)
            if len(run_data) == 0:
                continue
            all_tags = list(run_data.keys())
            all_data[exp_name].append(run_data)

    # Get table
    from pandas import DataFrame
    df = DataFrame(columns=all_tags)
    max_steps = {}
    for exp_name in exp_names:
        steps, mean_de_bleus, _ = combine_series([run_data[tags[0]] for run_data in all_data[exp_name]])
        max_id = np.argmax(mean_de_bleus)
        max_step = steps[max_id]
        max_steps[exp_name] = max_step
        print('{} max step: {}'.format(exp_name, max_step))
        for tag in all_tags:
            _, means, stds = combine_series([run_data[tag] for run_data in all_data[exp_name]])
            max_mean = means[max_id]
            max_std = stds[max_id]
            df.loc[exp_name, tag] = "{:.3f}({:.3f})".format(max_mean, max_std)
    print(df)

    # Start plotting
    if args.same_canvas:
        nb_col = 2
        nb_row = int((len(tags) + 1) / nb_col)
        matplotlib.rc('font', size=20)
        fig, axs = plt.subplots(nb_row, nb_col, figsize=(13*nb_col, 10*nb_row))
        for tag, ax in zip(all_tags, axs.reshape([-1])):
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
            plot_each_tag(all_data, ax, exp_names, max_steps, tag, font_size=10, use_median=args.use_median,
                          names=names, tags=tags)
        if args.use_median:
            fig.savefig(os.path.join(args.output_dir, 'result_median.png'))
        else:
            fig.savefig(os.path.join(args.output_dir, 'result_mean.png'))
    else:
        matplotlib.rc('font', size=20)
        for tag in all_tags:
            fig, ax = plt.subplots(figsize=(8, 7))
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
            plot_each_tag(all_data, ax, exp_names, max_steps, tag, font_size=20, use_median=args.use_median,
                          names=names, tags=tags)
            fig.savefig(os.path.join(args.output_dir, '{}.png'.format(names[tags.index(tag)])),
                        bbox_inches='tight')


if __name__ == '__main__':
    main()
