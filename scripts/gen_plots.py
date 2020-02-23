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
from math import ceil
import math

# The Tags that I care about
TAGS = ['eval_bleu_de', 'eval_bleu_en', 'eval_en_nll_lm', 'eval_r1_acc', 'eval_nll_real', 'dev_neg_Hs']
NAMES = ['BLEU_De', 'BLEU_En', 'NLL', 'R1', 'Real NLL', 'Neg Entropy']
# Plot Config
NB_COL = 2
NB_ROW = ceil(len(TAGS) / NB_COL)


class Series:
    def __init__(self):
        self.values = []
        self.steps = []

    def add(self, step, val):
        """ Insert step and value. Maintain sorted w.r.t. steps """
        if len(self.steps) == 0:
            self.steps.append(step)
            self.values.append(val)
        else:
            for idx in reversed(range(len(self.steps))):
                if step > self.steps[idx]:
                    break
            else:
                idx = -1
            self.steps.insert(idx + 1, step)
            self.values.insert(idx + 1, val)

    def verify(self):
        for i in range(len(self.steps) - 1):
            assert self.steps[i] <= self.steps[i + 1]


def combine_series(series_list):
    """
    :param series_list: a list of `Series` assuming steps are aligned
    :return: steps, means, stds
    """
    step_sizes = [len(series.steps) for series in series_list]
    min_idx = np.argmin(step_sizes)
    steps = series_list[min_idx].steps

    # [nb_run, nb_steps]
    all_values = [series.values[:len(steps)] for series in series_list]
    means = np.mean(all_values, axis=0)
    stds = np.std(all_values, axis=0)
    return steps, means, stds


def parse_tb_event_file(event_file):
    data = {}
    for e in tf.compat.v1.train.summary_iterator(event_file):
        for v in e.summary.value:
            tag = v.tag.replace('/', '_')
            if tag in TAGS:
                if data.get(tag) is None:
                    data[tag] = Series()
                if 'nll' in tag:
                    data[tag].add(step=e.step, val=np.abs(v.simple_value))
                else:
                    data[tag].add(step=e.step, val=v.simple_value)

    for tag in data:
        data[tag].verify()
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_dir')
    parser.add_argument('output_dir')
    args = parser.parse_args()

    exp_names = os.listdir(args.exp_dir)
    print('We have {} experiments'.format(len(exp_names)))

    all_data = {}
    all_tags = []
    for exp_name in exp_names:
        all_data[exp_name] = []
        runs = os.listdir(os.path.join(args.exp_dir, exp_name))
        print('We have {} runs for {}'.format(len(runs), exp_name))
        for run in runs:
            event_file = os.listdir(os.path.join(args.exp_dir, exp_name, run))[0]
            run_data = parse_tb_event_file(os.path.join(args.exp_dir, exp_name, run, event_file))
            if len(run_data) == 0:
                continue
            all_tags = list(run_data.keys())
            all_data[exp_name].append(run_data)


    # Get table
    from pandas import DataFrame
    df = DataFrame(columns=all_tags)
    max_steps = {}
    for exp_name in exp_names:
        steps, mean_de_bleus, _ = combine_series([run_data[TAGS[0]]
                                              for run_data in all_data[exp_name]])
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
    matplotlib.rc('font', size=20)
    for tag in all_tags:
        fig, ax = plt.subplots(figsize=(8, 7))
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        data = {}
        min_steps = math.inf
        for exp_name in exp_names:
            steps, means, stds = combine_series([run_data[tag] for run_data in all_data[exp_name]])
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
                #if step <= plot_steps:
                #    new_steps.append(step)
                #    new_means.append(mean)
                #    new_stds.append(std)
                #else:
                #    break
            new_means = np.array(new_means)
            new_stds = np.array(new_stds)
            line, = ax.plot(new_steps, new_means)
            line.set_label(exp_name)
            ax.fill_between(new_steps, new_means - new_stds, new_means + new_stds,
                            alpha=0.2)
        ax.set_xlabel('steps', fontsize=20)
        ax.set_title(NAMES[TAGS.index(tag)])
        ax.legend(fontsize=20)
        fig.savefig(os.path.join(args.output_dir, '{}.png'.format(NAMES[TAGS.index(tag)])))


if __name__ == '__main__':
    main()
