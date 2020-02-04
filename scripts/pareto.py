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
from itertools import product
import matplotlib.pyplot as plt
from math import ceil
import math

# The Tags that I care about
Y_TAGS = ['eval_bleu_de']
Y_NAMES = ['BLEU De']
X_TAGS = ['eval_bleu_en', 'eval_r1_acc', "eval_en_nll_lm"]
X_NAMES = ['BLEU En', 'R1', 'NLL']

# Plot Config
NB_PLOTS = len(Y_TAGS) * len(X_TAGS)
NB_COL = 2
NB_ROW = ceil(NB_PLOTS / NB_COL)


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
            if tag in X_TAGS + Y_TAGS:
                if data.get(tag) is None:
                    data[tag] = Series()
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

    # Start plotting
    fig, axs = plt.subplots(NB_ROW, NB_COL, figsize=(7*NB_COL, 5*NB_ROW))
    for idx, (x_tag, y_tag) in enumerate(product(X_TAGS, Y_TAGS)):
        ax = axs.reshape(-1)[idx]
        x_data = {}
        y_data = {}
        for exp_name in exp_names:
            _, means, stds = combine_series([run_data[x_tag] for run_data in all_data[exp_name]])
            x_data[exp_name] = (means, stds)
            _, means, stds = combine_series([run_data[y_tag] for run_data in all_data[exp_name]])
            y_data[exp_name] = (means, stds)

        # Use the smallest step in data
        for exp_name in exp_names:
            x_means, _ = x_data[exp_name]
            y_means, _ = y_data[exp_name]
            line, = ax.plot(x_means, y_means, 'x')
            line.set_label(exp_name)
        ax.set_xlabel(X_NAMES[X_TAGS.index(x_tag)])
        ax.set_ylabel(Y_NAMES[Y_TAGS.index(y_tag)])
    axs.reshape(-1)[0].legend()
    fig.savefig(os.path.join(args.output_dir, 'output.png'))


if __name__ == '__main__':
    main()
