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

# The Tags that I care about
TAGS = ['grad_cosine', 'grad_rl_grad_norm', 'grad_s2p_grad_norm']

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
    parser.add_argument('-event_file')
    parser.add_argument('-out')
    args = parser.parse_args()
    run_data = parse_tb_event_file(args.event_file)

    matplotlib.rc('font', size=20)
    # Plot cosine
    cosine_data = run_data['grad_cosine']
    fig, ax = plt.subplots(figsize=(8, 7))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.plot(cosine_data.steps, cosine_data.values)
    fig.savefig(os.path.join(args.out, 'cosine.png'))

    # Plot to norm
    fig, ax = plt.subplots(figsize=(8, 7))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    sp_norm = run_data['grad_rl_grad_norm']
    su_norm = run_data['grad_s2p_grad_norm']
    line, = ax.plot(sp_norm.steps, sp_norm.values)
    line.set_label('SP Grad Norm')
    line, = ax.plot(su_norm.steps, su_norm.values)
    line.set_label('SU Grad Norm')
    ax.legend(fontsize=20)
    fig.savefig(os.path.join(args.out, 'grad_norms.png'))

if __name__ == '__main__':
    main()
