"""
A result of EOT run has folder structure
itlearn_eot
├── iters_0_run_2
│   └── logs
│       └── events.out.tfevents.1586540408.eos14.server.mila.quebec
├── iters_100000_run_1
│   └── logs
│       └── events.out.tfevents.1586579676.eos14.server.mila.quebec
├── iters_10000_run_3
│   └── logs
│       └── events.out.tfevents.1586545222.eos14.server.mila.quebec
...
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import argparse
import os
import tensorflow as tf
from tqdm import tqdm


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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-input_dir', required=True)
    return parser.parse_args()


def parse_tb_event_file(event_file, tags):
    data = {}
    for e in tf.compat.v1.train.summary_iterator(event_file):
        for v in e.summary.value:
            tag = v.tag
            if tag in tags:
                if data.get(tag) is None:
                    data[tag] = Series()
                data[tag].add(step=e.step, val=v.simple_value)

    for tag in data:
        data[tag].verify()
    return data


def plot_learning_curves(input_dir):
    tags = ['bleu/bleu', 'dev/nll', 'train/nll']
    event_dirs = {(int(fname.split('_')[1]), int(fname.split('_')[-1])): fname
                  for fname in os.listdir(input_dir) if 'iters_' in fname and
                  os.path.isdir(os.path.join(input_dir, fname))}
    data = {}
    for (iters, runs), event_dir in event_dirs.items():
        log_folder = os.path.join(input_dir, event_dir, 'logs')
        event_file = os.listdir(log_folder)[0]
        series = parse_tb_event_file(os.path.join(log_folder, event_file),
                                     tags=tags)
        if data.get(iters, None) is not None:
            data[iters].append(series)
        else:
            data[iters] = [series]

    print('Start plotting...')
    matplotlib.rc('font', size=20)
    for iters in tqdm(data):
        nb_col = 2
        nb_row = int((len(tags) + 1) / 2)
        fig, axs = plt.subplots(nb_row, nb_col, figsize=(8 * nb_row, 10 * nb_col))
        all_run_data = data[iters]

        for tag, ax in zip(tags, axs.reshape(-1)):
            series_list = [run_data[tag] for run_data in all_run_data]
            steps, means, stds = combine_series(series_list)
            ax.plot(steps, means)
            ax.fill_between(steps, means - stds, means + stds,
                            alpha=0.2)
            ax.set_xlabel('steps', fontsize=20)
            ax.set_title(tag, fontsize=20)
        fig.savefig(os.path.join(input_dir, 'lc_iters_{}.png'.format(iters)))


if __name__ == '__main__':
    args = get_args()
    plot_learning_curves(args.input_dir)
