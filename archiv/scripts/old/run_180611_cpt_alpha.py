import random
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

jobname = "180611_cpt_alpha"
mem = 16
hours = 24
slurm_dir = home_dir + "180611_cpt_alpha" if not interactive else ""
if not interactive and not os.path.exists(slurm_dir):
    os.mkdir(slurm_dir)
event_path = "event_180611_cpt_alpha"
model_path = "model_180611_cpt_alpha"
log_path = "log_180611_cpt_alpha"
decoding_path = "decoding_180611_cpt_alpha"

cpt_iters = [200, 400, 600, 800, 1000, 2500, 5000, 10000]
msg_len_ratios = [0.25, 0.5, 1.0, 1.5, 2.0]
lrs = [3e-4, 1e-4, 3e-5] # 2

nll_reward = "mean"

ce_coeff, pg_coeff, vf_coeff = (1.0, 10.0, 1.0)

lr_anneal, linear_steps = "linear", 100000

drop_ratio = 0.3
eval_every = 200
save_every = -1
max_steps = 1000000
job_idx = 0
grad_clip = 5.0

shuffle = True

cmds = []
for cpt_iter in cpt_iters: # 8
    for msg_len_ratio in msg_len_ratios: # 5
        for lr in lrs: # 3

            job_idx += 1

            options = "sbatch -J {}_{} --gres=gpu:1 --mem {}GB --time={}:00:00 --output={}/%j.out".format(jobname, job_idx, mem, hours, slurm_dir)

            cmd = "python {} \
                    --load_vocab --load_dataset \
                    --model RNNAttn\
                    --setup joint\
                    --nll_reward {}\
                    --send_method reinforce\
                    --ce_coeff {} --pg_coeff {} --vf_coeff {}\
                    --drop_ratio {}\
                    --grad_clip {}\
                    --lr {} --lr_anneal {} --linear_anneal_steps {} --max_training_steps {}\
                    --cpt_iter {}\
                    --eval_every {} --save_every {}\
                    --msg_len_ratio {}\
                    --event_path {} --log_path {} --decoding_path {} --model_path {}".format(run, 
                        nll_reward, 
                        ce_coeff, pg_coeff, vf_coeff, 
                        drop_ratio, 
                        grad_clip, 
                        lr, lr_anneal, linear_steps, max_steps, 
                        cpt_iter, 
                        eval_every, save_every, 
                        msg_len_ratio, 
                        event_path, log_path, decoding_path, model_path)

            if not interactive:
                cmd += " --no_tqdm"

            cmd = re.sub( '\s+', ' ', cmd ).strip()
            #print (cmd, "\n")

            cmd = '{} {} \"{}\"'.format(options, slurm_file, cmd)
            cmds.append(cmd)

if shuffle:
    random.shuffle(cmds)

for cmd in cmds:
    subprocess.call(cmd, shell=True)
    time.sleep(0.001)
print ("Submitted {} jobs".format(job_idx))
