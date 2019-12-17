import argparse
import random
import sys
import time
import os
import re
import subprocess
import pprint

parser = argparse.ArgumentParser()
parser.add_argument('--interactive', action='store_true', default=False, help='save a pre-processed dataset')
parser.add_argument('--shuffle', action='store_true', default=False, help='save a pre-processed dataset')
parser.add_argument('--nosubmit', action='store_true', default=False, help='save a pre-processed dataset')
parser.add_argument('--print', action='store_true', default=False, help='save a pre-processed dataset')
args = parser.parse_args()

slurm_file = "/private/home/jasonleeinf/dotfiles/run/run_learnfair.sh"
run = "/private/home/jasonleeinf/scratch/groundcomms/src/run.py"
home_dir = "/private/home/jasonleeinf/slurm/"

jobname = "run_180613_rerun_rnn"
mem = 24
hours = 12
slurm_dir = home_dir + "180613_rerun_rnn" if not args.interactive else ""
if not args.interactive and not os.path.exists(slurm_dir):
    os.mkdir(slurm_dir)
event_path = "event_180613_rerun_rnn"
model_path = "model_180613_rerun_rnn"
log_path = "log_180613_rerun_rnn"
decoding_path = "decoding_180613_rerun_rnn"

pairs = ["fr_en", "en_de"]
drop_ratios = [0.2, 0.3, 0.4, 0.5]
lrs = [1e-3, 3e-4, 1e-4] # 3
save_every = 100
eval_every = 100

lr1 = ["half", "linear"] # 2
lr2 = [0, 30000]
lr_anneals = [(a,b) for (a,b) in zip(lr1, lr2)]
max_steps = 100000

cmds = []
job_idx = 0
for pair in pairs: # 2
    for drop_ratio in drop_ratios: # 4
        for lr in lrs: # 3 
            for (lr_anneal, linear_steps) in lr_anneals: # 3
                job_idx += 1

                options = "sbatch -J {}_{} --gres=gpu:1 --mem {}GB --time={}:00:00 --output={}/%j.out".format(jobname, job_idx, mem, hours, slurm_dir)

                cmd = "python {} \
                        --load_vocab --load_dataset \
                        --setup single\
                        --pair {}\
                        --model RNN\
                        --eval_every {}\
                        --save_every {}\
                        --drop_ratio {}\
                        --lr {}\
                        --lr_anneal {}\
                        --linear_anneal_steps {}\
                        --max_training_steps {}\
                        --event_path {} \
                        --log_path {} \
                        --model_path {}".format(run, pair, eval_every, save_every, drop_ratio, lr, lr_anneal, linear_steps, max_steps, event_path, log_path, model_path)

                if not args.interactive:
                    cmd += " --no_tqdm"

                cmd = re.sub( '\s+', ' ', cmd ).strip()
                if args.print:
                    print (cmd, "\n")

                cmd = '{} {} \"{}\"'.format(options, slurm_file, cmd)
                cmds.append(cmd)

if args.shuffle:
    random.shuffle(cmds)

if not args.nosubmit:
    for cmd in cmds:
        subprocess.call(cmd, shell=True)
        time.sleep(0.001)

print ("Submitted {} jobs".format(job_idx))
