import os

import numpy as np
import tensorflow as tf


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


def parse_tb_event_files(event_dir, tags):
    data = {}
    event_files = [os.path.join(event_dir, fname) for fname in os.listdir(event_dir)
                   if 'events.' in fname and not os.path.isdir(fname)]
    print('Found {} event file'.format(len(event_files)))
    for event_file in event_files:
        for e in tf.compat.v1.train.summary_iterator(event_file):
            for v in e.summary.value:
                tag = v.tag.replace('/', '_')
                if tag in tags:
                    if data.get(tag) is None:
                        data[tag] = Series()
                    data[tag].add(step=e.step, val=v.simple_value)

    for tag in data:
        data[tag].verify()
    return data


def combine_series(series_list, use_median=False):
    """
    :param series_list: a list of `Series` assuming steps are aligned
    :return: steps, values, stds
    """
    step_sizes = [len(series.steps) for series in series_list]
    min_idx = np.argmin(step_sizes)
    steps = series_list[min_idx].steps

    # [nb_run, nb_steps]
    all_values = [series.values[:len(steps)] for series in series_list]
    if not use_median:
        values = np.mean(all_values, axis=0)
    else:
        values = np.median(all_values, axis=0)
    stds = np.std(all_values, axis=0)
    return steps, values, stds