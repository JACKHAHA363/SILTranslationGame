"""
Do Grid Search
"""
import argparse
import json
from itertools import product
import itlearn
import os
import sys
from subprocess import call

PROJ_PATH = os.path.dirname(itlearn.__file__)
SCRIPT_PATH = os.path.join(PROJ_PATH, 'train.py')
PYBIN = sys.executable
SLURM_FILE = os.path.join(os.path.dirname(PROJ_PATH), 'scripts', 'run_graham.sh')
print('train_script', SCRIPT_PATH)
print('python bin', PYBIN)
print('slurm run file', SLURM_FILE)
EXCLUDES = ['save_at']

SLURM_OPTION = ['sbatch', SLURM_FILE]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--exp_dir', required=True)
    parser.add_argument('--sweep', required=True, help='path to json of sweep')
    parser.add_argument('--test', action='store_true', help='not calling just printing')
    return parser.parse_args()


def main():
    args = get_args()
    with open(args.sweep, 'r') as f:
        sweep_json = json.load(f)
    all_keys = list(sweep_json.keys())
    sweep_keys = [k for k in all_keys if isinstance(sweep_json[k], list) and k not in EXCLUDES]
    non_sweep_keys = [k for k in all_keys if k not in sweep_keys]
    print('Sweeping ', sweep_keys)

    # Command with fixed args
    cmd = [PYBIN, SCRIPT_PATH,
           '--data_dir', os.path.abspath(args.data_dir),
           '--exp_dir', os.path.abspath(args.exp_dir),
           '--config', os.path.abspath(args.sweep)]
    for k in non_sweep_keys:
        cmd += ['--{}'.format(k), str(sweep_json[k])]
    for sweep_args in product(*[sweep_json[k] for k in sweep_keys]):
        non_fixed = []
        for k, val in zip(sweep_keys, sweep_args):
            non_fixed += ['--{}'.format(k), str(val)]
        final_cmd = SLURM_OPTION + cmd + non_fixed
        print(final_cmd)
        if not args.test:
            call(final_cmd)


if __name__ == '__main__':
    main()
