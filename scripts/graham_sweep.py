"""
Do Grid Search
"""
import argparse
import json
from itertools import product

PYBIN = ""
SCRIPT_PATH = ""

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--exp_dir', required=True)
    parser.add_argument('--sweep', required=True, help='path to json of sweep')
    return parser.parse_args()


def main():
    args = get_args()
    with open(args.sweep, 'r') as f:
        sweep_json = json.load(f)
    all_keys = list(sweep_json.keys())
    sweep_keys = [k for k in all_keys if isinstance(sweep_json[k], list)]
    non_sweep_keys = [k for k in all_keys if k not in sweep_keys]
    print('Sweeping ', sweep_keys)

    # Command with fixed args
    cmd = [PYBIN, SCRIPT_PATH,
           '--data_dir', args.data_dir,
           '--exp_dir', args.exp_dir,
           '--config', args.sweep]
    for k in non_sweep_keys:
        cmd += ['--{}'.format(k), str(sweep_json[k])]

    for sweep_args in product(*[sweep_json[k] for k in sweep_keys]):
        non_fixed = []
        for k, val in zip(sweep_keys, sweep_args):
            non_fixed += ['--{}'.format(k), str(val)]
        final_cmd = cmd + non_fixed
        print(final_cmd)


if __name__ == '__main__':
    main()
