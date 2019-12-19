import sys
import time
import os
import re
import subprocess
import pprint

interactive = False

slurm_file = "/private/home/jasonleeinf/dotfiles/run/run_learnfair.sh"
run = "/private/home/jasonleeinf/scratch/groundcomms/src/run.py"
home_dir = "/private/home/jasonleeinf/slurm/"

jobname = "180603"
mem = 24
hours = 12
slurm_dir = home_dir + "180603_2" if not interactive else ""
if not interactive and not os.path.exists(slurm_dir):
    os.mkdir(slurm_dir)
event_path = "event_180603_2"
model_path = "model_180603_2"
log_path = "log_180603_2"

send_methods = ["gumbel", "reinforce"] # 2
#cpt_iters = [500,1500,2500,3500] # 4
cpt_iters = [4500,5500] # 4

drop_ratios = [0.2, 0.3] # 2

lrs = [3e-4, 1e-4, 3e-5] # 3
lr1 = ["half", "linear"] # 2
lr2 = [0, 20000]
lr_anneals = [(a,b) for (a,b) in zip(lr1, lr2)]

eval_every = 200
save_every = -1
max_steps = 50000
job_idx = 0

for send_method in send_methods:
    for cpt_iter in cpt_iters:
        for lr in lrs:
            for drop_ratio in drop_ratios:
                for (lr_anneal, linear_steps) in lr_anneals:
                    job_idx += 1

                    options = "sbatch -J {}_{} --gres=gpu:1 --mem {}GB --time={}:00:00 --output={}/%j.out".format(jobname, job_idx, mem, hours, slurm_dir)

                    cmd = "python {} \
                            --load_vocab --load_dataset \
                            --finetune\
                            --setup joint\
                            --model RNNAttn\
                            --send_method {}\
                            --cpt_iter {}\
                            --eval_every {}\
                            --save_every {}\
                            --drop_ratio {}\
                            --lr {}\
                            --lr_anneal {}\
                            --linear_anneal_steps {}\
                            --max_training_steps {}\
                            --event_path {} \
                            --log_path {} \
                            --model_path {}".format(run, send_method, cpt_iter, eval_every, save_every, drop_ratio, lr, lr_anneal, linear_steps, max_steps, event_path, log_path, model_path)

                    #cmd += " --early_stop"

                    if not interactive:
                        cmd += " --no_tqdm"

                    cmd = re.sub( '\s+', ' ', cmd ).strip()
                    #print (cmd, "\n")

                    cmd = '{} {} \"{}\"'.format(options, slurm_file, cmd)
                    #print (cmd, "\n")
                    subprocess.call(cmd, shell=True)
                    time.sleep(0.01)

