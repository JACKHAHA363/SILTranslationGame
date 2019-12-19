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

jobname = "180609"
mem = 16
hours = 24
slurm_dir = home_dir + "180609" if not interactive else ""
if not interactive and not os.path.exists(slurm_dir):
    os.mkdir(slurm_dir)
event_path = "event_180609"
model_path = "model_180609"
log_path = "log_180609"
decoding_path = "decoding_180609"

coeffs = [ (1.0, 10.0, 0.01), # 2
           (1.0, 100.0, 0.01),
           (1.0, 10.0, 0.1),
           (1.0, 100.0, 0.1),
         ]

msg_len_ratios = [1.0] # 5

lr1 = ["linear", "linear"] # 2
lr2 = [20000, 100000]
lr_anneals = [(a,b) for (a,b) in zip(lr1, lr2)]

lrs = [1e-4, 3e-5] # 2

cpt_iters = [2500,3000] # 3
#cpt_iters = [1500,2000] # 3
#cpt_iters = [500,1000] # 3

drop_ratio = 0.3
eval_every = 200
save_every = -1
max_steps = 300000
job_idx = 0
grad_clip = 5

for (ce_coeff, pg_coeff, vf_coeff) in coeffs: # 4
    for msg_len_ratio in msg_len_ratios: # 1
        for (lr_anneal, linear_steps) in lr_anneals: # 2
            for lr in lrs: # 2
                for cpt_iter in cpt_iters: # 2
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
                            --msg_len_ratio {}\
                            --event_path {} \
                            --log_path {} \
                            --decoding_path {} \
                            --model_path {}".format(run, ce_coeff, pg_coeff, vf_coeff, cpt_iter, eval_every, save_every, drop_ratio, lr, lr_anneal, linear_steps, max_steps, grad_clip, msg_len_ratio, event_path, log_path, decoding_path, model_path)

                    cmd += " --use_en_lm"

                    if not interactive:
                        cmd += " --no_tqdm"

                    cmd = re.sub( '\s+', ' ', cmd ).strip()
                    #print (cmd, "\n")

                    cmd = '{} {} \"{}\"'.format(options, slurm_file, cmd)
                    cmd = "cd /private/home/jasonleeinf/scratch/groundcomms/src; " + cmd
                    subprocess.call(cmd, shell=True)

                    #sys.exit()
                    time.sleep(0.001)

