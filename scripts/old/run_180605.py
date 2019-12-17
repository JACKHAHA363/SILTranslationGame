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

jobname = "180605_2"
mem = 16
hours = 15
slurm_dir = home_dir + "180605_2" if not interactive else ""
if not interactive and not os.path.exists(slurm_dir):
    os.mkdir(slurm_dir)
event_path = "event_180605_2"
model_path = "model_180605_2"
log_path = "log_180605_2"
decoding_path = "decoding_180605_2"

send_methods = ["reinforce"]
cpt_iters = [500,1500,2500,3500] # 4

drop_ratios = [0.2, 0.3] # 2

coeffs = [ (1.0, 10.0, 0.01), # 6
           (1.0, 10.0, 0.1),
           (1.0, 10.0, 1.0),
           (1.0, 100.0, 0.01),
           (1.0, 100.0, 0.1),
           (1.0, 100.0, 1.0),
         ]

lrs = [3e-4, 1e-4, 3e-5] # 3
lr1 = ["half", "linear"] # 2
lr2 = [0, 20000]
lr_anneals = [(a,b) for (a,b) in zip(lr1, lr2)]

eval_every = 200
save_every = -1
max_steps = 50000
job_idx = 0
grad_clip = 5

for drop_ratio in drop_ratios:
    for (lr_anneal, linear_steps) in lr_anneals:
        for lr in lrs:
            for cpt_iter in cpt_iters:
                for (ce_coeff, pg_coeff, vf_coeff) in coeffs:
                    job_idx += 1

                    options = "sbatch -J {}_{} --gres=gpu:1 --mem {}GB --time={}:00:00 --output={}/%j.out".format(jobname, job_idx, mem, hours, slurm_dir)

                    cmd = "python {} \
                            --ce_coeff {} --pg_coeff {} --vf_coeff {}\
                            --load_vocab --load_dataset \
                            --finetune\
                            --setup joint\
                            --model RNNAttn\
                            --send_method reinforce\
                            --cpt_iter {}\
                            --eval_every {}\
                            --save_every {}\
                            --drop_ratio {}\
                            --lr {}\
                            --lr_anneal {}\
                            --linear_anneal_steps {}\
                            --max_training_steps {}\
                            --grad_clip {}\
                            --event_path {} \
                            --log_path {} \
                            --decoding_path {} \
                            --model_path {}".format(run, ce_coeff, pg_coeff, vf_coeff, cpt_iter, eval_every, save_every, drop_ratio, lr, lr_anneal, linear_steps, max_steps, grad_clip, event_path, log_path, decoding_path, model_path)

                    #cmd += " --early_stop"

                    if not interactive:
                        cmd += " --no_tqdm"

                    cmd = re.sub( '\s+', ' ', cmd ).strip()
                    #print (cmd, "\n")

                    cmd = '{} {} \"{}\"'.format(options, slurm_file, cmd)
                    subprocess.call(cmd, shell=True)
                    time.sleep(0.001)

