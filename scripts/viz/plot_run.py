"""
Given a tensorboard event and name of tags. Plot all the scalars
"""
import argparse
import os
import matplotlib.pyplot as plt
import matplotlib

from itlearn.utils.viz import parse_tb_event_files


def plot_tag(ax, tag, run_data):
    data = run_data[tag]
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.plot(data.steps, data.values)
    ax.set_xlabel('steps', fontsize=20)
    ax.set_title(tag)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-event_dir')
    parser.add_argument('-tags', nargs='+')
    parser.add_argument('-same_canvas', action='store_true')
    args = parser.parse_args()
    print('Plot:', args.tags)
    run_data = parse_tb_event_files(args.event_dir, args.tags)

    matplotlib.rc('font', size=20)
    if args.same_canvas:
        nb_col = 2
        nb_row = int((len(args.tags) + 1) / nb_col)
        fig, axs = plt.subplots(nb_row, nb_col, figsize=(13 * nb_col, 10 * nb_row))
        for tag, ax in zip(args.tags, axs.reshape(-1)):
            plot_tag(ax, tag, run_data)
        fig.savefig(os.path.join(args.event_dir, 'result.png'))
    else:
        for tag in args.tags:
            fig, ax = plt.subplots(figsize=(8, 7))
            plot_tag(ax, tag, run_data)
            fig.savefig(os.path.join(args.event_dir, '{}.png'.format(tag)))



if __name__ == '__main__':
    main()
