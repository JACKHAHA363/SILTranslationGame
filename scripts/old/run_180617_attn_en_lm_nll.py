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

jobname = "180617_attn_en_lm_nll"
mem = 20
hours = 36
slurm_dir = home_dir + "180617_attn_en_lm_nll" if not args.interactive else ""
if not args.interactive and not os.path.exists(slurm_dir):
    os.mkdir(slurm_dir)
event_path = "event_180617_attn_en_lm_nll"
model_path = "model_180617_attn_en_lm_nll"
log_path = "log_180617_attn_en_lm_nll"
decoding_path = "decoding_180617_attn_en_lm_nll"

models = ["RNN", "RNNAttn"]
cpt_iters = { "RNNAttn" : [200, 400, 600, 800, 1000, 2500, 5000, 10000],
              "RNN" : [1000, 2000, 3000, 4000, 5000, 10000] }
msg_len_ratio = 1.0
lrs = [3e-4, 1e-4, 3e-5] # 2

nll_reward = "mean"

ce_de_coeff, pg_de_coeff, vf_de_coeff = (1.0, 10.0, 1.0)

lr_anneal, linear_steps = "linear", 100000

drop_ratio = 0.3
eval_every = 200
save_every = -1
max_steps = 1000000
job_idx = 0
grad_clip = 5.0

cmds = []
for model in models:
    for cpt_iter in cpt_iters[model]: # 8
        for lr in lrs: # 3

            job_idx += 1

            options = "sbatch -J {}_{} --gres=gpu:1 --mem {}GB --time={}:00:00 --output={}/%j.out".format(jobname, job_idx, mem, hours, slurm_dir)

            cmd = "python {} \
                    --load_vocab --load_dataset \
                    --model {}\
                    --setup joint\
                    --nll_reward {}\
                    --send_method reinforce\
                    --ce_de_coeff {} --pg_de_coeff {} --vf_de_coeff {}\
                    --drop_ratio {}\
                    --grad_clip {}\
                    --lr {} --lr_anneal {} --linear_anneal_steps {} --max_training_steps {}\
                    --cpt_iter {}\
                    --eval_every {} --save_every {}\
                    --msg_len_ratio {}\
                    --event_path {} --log_path {} --decoding_path {} --model_path {}".format(run, 
                        model,
                        nll_reward, 
                        ce_de_coeff, pg_de_coeff, vf_de_coeff, 
                        drop_ratio, 
                        grad_clip, 
                        lr, lr_anneal, linear_steps, max_steps, 
                        cpt_iter, 
                        eval_every, save_every, 
                        msg_len_ratio, 
                        event_path, log_path, decoding_path, model_path)

            cmd += " --use_en_lm"
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
