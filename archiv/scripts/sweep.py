import ipdb
import itertools
from time import localtime, strftime
import argparse
import random
import sys
import time
import os
import re
import subprocess
import pprint
import json

def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    vals = [v if isinstance(v, list) else [v] for v in vals]
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))

def split_tuple(dic):
    newdic = {}
    for k, v in dic.items():
        if " " in k:
            assert (isinstance(v, list))
            newdic.update({kk:vv for (kk, vv) in zip(k.split(), v)})
        else:
            newdic[k]=v
    return newdic

def argfy(param):
    return " ".join(["--{} {}".format(k,v) for (k,v) in param.items()])

parser = argparse.ArgumentParser()
parser.add_argument('--json',       type=str)
parser.add_argument('--shuffle',    action='store_true',  default=False)
parser.add_argument('--test',       action='store_true',  default=False)
parser.add_argument('--queue',      type=str,             choices=['priority', 'learnfair', 'uninterrupted', 'dev'])
parser.add_argument('--memory',     type=int,             default=20)
parser.add_argument('--hours',      type=int,             default=12)
parser.add_argument('--cpus',       type=int,             default=4)
args = parser.parse_args()

date = strftime("%y%m%d", localtime())

params = json.load( open( args.json ) )
args.gpu = True
experiment = params["experiment"]
params["date"] = date
params = list(product_dict(**params))
params = [split_tuple(p) for p in params]
params = [argfy(p) for p in params]

slurm_file = "/private/home/jasonleeinf/dotfiles/run/run_learnfair_{}.sh".format(\
             "gpu" if args.gpu else "cpu")
run = "/private/home/jasonleeinf/scratch/groundcomms/src/run.py"
slurm_dir = "/private/home/jasonleeinf/slurm/groundcomms/"
slurm_dir = slurm_dir + "{}_{}".format(date, experiment)
python = "python_ipdb" if args.test else "python"
if not args.test and not os.path.exists(slurm_dir):
    os.mkdir(slurm_dir)

jobname = "{}_{}".format(date, experiment)
queue = args.queue

cmds = []
job_idx = 0
for idx, param in enumerate(params):
    if queue == "dev" and idx >= 16:
        queue = "priority"
    slurm_options = "sbatch -J {} --partition={} --mem {}GB --cpus-per-task {} --time={}:00:00 --output={}/%j.out".format(jobname, queue, args.memory, args.cpus, args.hours, slurm_dir)
    if args.gpu:
        slurm_options += " --gres=gpu:1"

    job_idx += 1
    cmd = "{} {} {}".format(python, run, param)
    if args.test:
        cmd = cmd + " --debug True"
    cmd = "cd {}; ".format(run[:run.rfind("/")]) + cmd
    if args.test:
        print (cmd, "\n")
    cmd = '{} {} \"{}\"'.format(slurm_options, slurm_file, cmd)
    cmds.append(cmd)

if args.shuffle:
    random.shuffle(cmds)

if not args.test:
    for cmd in cmds:
        subprocess.call(cmd, shell=True)
        time.sleep(0.01)
    print ("Submitted {} jobs".format(job_idx))
else:
    print ("Printing {} jobs".format(job_idx))

