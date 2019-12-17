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

"""
lm_hps = [ "--data multi30k --arch Feedforward --lr 5.0 --dropout 0.5 --filter_width 4",
 "--data multi30k --arch RNN --lr 20.0 --dropout 0.5",
 "--data coco --arch Feedforward --lr 20.0 --dropout 0.5 --filter_width 4",
 "--data coco --arch RNN --lr 10.0 --dropout 0.5",
 "--data wikitext2 --arch Feedforward --lr 5.0 --dropout 0.3 --filter_width 4",
 "--data wikitext2 --arch RNN --lr 20.0 --dropout 0.5" ]
"""

lm_hps = ['--data multi30k --arch RNN --emsize 400 --nhid 400 --dropout 0.5']

gc_hps = [ \
          (400, 1.0, 3e-05), \
           (400, 1.0, 1e-04), \
           (600, 1.0, 3e-05), \
           (800, 1.0, 1e-04), \
           (1000, 1.0, 1e-04), \
           (2500, 1.0, 1e-04), \
           (2500, 2.0, 3e-05), \
           (2500, 2.0, 3e-04), \
           (5000, 1.0, 1e-04), \
           (5000, 2.0, 3e-04), \
           (10000, 1.0, 1e-04), \
           (10000, 2.0, 3e-04), \
         ]
"""

gc_hps = [ \
         ]
"""

slurm_file = "/private/home/jasonleeinf/dotfiles/run/run_learnfair.sh"
run = "/private/home/jasonleeinf/scratch/groundcomms/src/lm/eval_msg.py"
home_dir = "/private/home/jasonleeinf/slurm/"

jobname = "run_180615_lm_nll"
mem = 24
hours = 12
slurm_dir = home_dir + "180615_lm_nll" if not args.interactive else ""
if not args.interactive and not os.path.exists(slurm_dir):
    os.mkdir(slurm_dir)

cmds = []
job_idx = 0
for lm_hp in lm_hps:
    for cpt, alpha, lr in gc_hps:
        job_idx += 1
        path = Path("/checkpoint/jasonleeinf/groundcomms/decoding_180612_cpt_alpha/multi30k")

        models = []
        for subdir in path.iterdir():
            ss = str(subdir)
            if "cpt{}_".format(cpt) in ss and \
               "msg{}x_".format(alpha) in ss and \
               "lr{:.0e}_".format(lr) in ss:
                models.append( ss )

        assert (len(models) == 1)
        # python_ipdb eval_msg.py --data multi30k --arch Feedforward --lr 5 --dropout 0.5 --filter 4 --msg_path 06.12_13.47.joint_cpt200_nllmean_ce1.0_pg10.0_vf1.0_lr3e-05_linear_ann100k_drop0.3_reinforce_msg2.0x_clip5.0_ --cuda

        options = "sbatch -J {}_{} --gres=gpu:1 --mem {}GB --time={}:00:00 --output={}/%j.out".format(jobname, job_idx, mem, hours, slurm_dir)
        cmd = "python {} --cuda {} --msg_path {}".format(run, lm_hp, models[0])
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

