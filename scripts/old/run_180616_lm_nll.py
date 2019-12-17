import ipdb
import argparse
import random
import sys
import time
import os
import re
import subprocess
import pprint
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--interactive', action='store_true', default=False, help='save a pre-processed dataset')
parser.add_argument('--shuffle', action='store_true', default=False, help='save a pre-processed dataset')
parser.add_argument('--nosubmit', action='store_true', default=False, help='save a pre-processed dataset')
parser.add_argument('--print', action='store_true', default=False, help='save a pre-processed dataset')
args = parser.parse_args()

sizes = [(600, 600), (400, 400)]
batch_size = 128
dropouts = [0.3,0.4,0.5]
tied = True
vocab_sizes = [5000, 7500, 10000, 15000, 17500, 20000]
lrs = [20, 10, 5]

slurm_file = "/private/home/jasonleeinf/dotfiles/run/run_learnfair.sh"
run = "/private/home/jasonleeinf/scratch/groundcomms/src/lm/main.py"
home_dir = "/private/home/jasonleeinf/slurm/"

jobname = "run_180616_lm_nll"
mem = 24
hours = 12
slurm_dir = home_dir + "180616_lm_nll" if not args.interactive else ""
if not args.interactive and not os.path.exists(slurm_dir):
    os.mkdir(slurm_dir)

cmds = []
job_idx = 0
for (emsize, nhid) in sizes: # 2
    for dropout in dropouts: # 3
        for vocab_size in vocab_sizes: # 6
            for lr in lrs: # 3

                job_idx += 1

                options = "sbatch -J {}_{} --gres=gpu:1 --mem {}GB --time={}:00:00 --output={}/%j.out".format(jobname, job_idx, mem, hours, slurm_dir)
                cmd = "python {} --data cocomulti --emsize {} --nhid {} --lr {} --batch_size {} --dropout {} --tied --cuda --vocab_size {}".format(run, emsize, nhid, lr, batch_size, dropout, vocab_size)

                if args.print:
                    print (cmd, "\n")
                cmd = "cd /private/home/jasonleeinf/scratch/groundcomms/src/lm; " + cmd

                cmd = '{} {} \"{}\"'.format(options, slurm_file, cmd)
                cmds.append(cmd)

if args.shuffle:
    random.shuffle(cmds)

if not args.nosubmit:
    for cmd in cmds:
        subprocess.call(cmd, shell=True)
        time.sleep(0.001)

print ("Submitted {} jobs".format(job_idx))

