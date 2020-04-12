"""
Folder Structure
"""
import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-paths', nargs='+', required=True)
    parser.add_argument('-names', nargs='+', required=True)
    parser.add_argument('-human_path')
    parser.add_argument('-out', default='eot_compare.png')
    return parser.parse_args()


def main():
    args = get_args()
    assert len(args.paths) == len(args.names)
    all_data = {name: pickle.load(open(path, 'rb')) for name, path in zip(args.names, args.paths)}
    human_data = None
    if args.human_path is not None:
        human_data = pickle.load(open(args.human_path, 'rb'))

    tags = list(all_data[args.names[0]]['statss'][0][0].keys())
    print('Start plotting...')
    NB_COL = 2
    NB_ROW = int((len(tags) + 1) / 2)
    fig, axs = plt.subplots(NB_ROW, NB_COL, figsize=(8 * NB_ROW, 10 * NB_COL))
    for key, ax in zip(tags, axs.reshape(-1)):
        min_steps = 0
        max_steps = 0
        for name in all_data:
            steps = all_data[name]['steps']
            if min(steps) <= min_steps:
                min_steps = min(steps)
            if max(steps) > max_steps:
                max_steps = max(steps)

            statss = all_data[name]['statss']
            if key == 'dev/nll - train/nll':
                values = [[stats['dev/nll'] - stats['train/nll'] for stats in run_stats] for run_stats in statss]
            else:
                values = [[stats[key] for stats in run_stats] for run_stats in statss]
            means = np.mean(values, -1)
            stds = np.std(values, -1)
            line, = ax.plot(steps, means)
            line.set_label(name)
            ax.fill_between(steps, means - stds, means + stds,
                            alpha=0.2)

        if human_data is not None:
            statss = human_data['human_stats']
            steps = [min_steps, max_steps]
            values = [[stats[key] for stats in statss] for _ in range(2)]
            means = np.mean(values, -1)
            stds = np.std(values, -1)
            line, = ax.plot(steps, means)
            line.set_label('human')
            ax.fill_between(steps, means - stds, means + stds,
                            alpha=0.2)

        ax.set_xlabel('steps', fontsize=15)
        ax.set_title(key)
        ax.legend()
    fig.savefig(args.out)


if __name__ == '__main__':
    main()
