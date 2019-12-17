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

jobname = "180606_2"
mem = 16
hours = 15
slurm_dir = home_dir + "180606_2" if not interactive else ""
if not interactive and not os.path.exists(slurm_dir):
    os.mkdir(slurm_dir)
event_path = "event_180606_2"
model_path = "model_180606_2"
log_path = "log_180606_2"
decoding_path = "decoding_180606_2"

cpt_iters = [1500,2500,3500,500] # 4

drop_ratio = 0.3 # 1

coeffs = [ (1.0, 10.0, 0.01), # 4
           (1.0, 100.0, 0.01),
           (1.0, 10.0, 0.1),
           (1.0, 100.0, 0.1),
         ]

msg_len_ratios = [0.25, 0.5, 0.75, 1.0, 1.5] # 5

lr = 1e-4 # 1
lr1 = ["linear", "linear"] # 2
lr2 = [20000, 30000]
lr_anneals = [(a,b) for (a,b) in zip(lr1, lr2)]

eval_every = 200
save_every = -1
max_steps = 50000
job_idx = 0
grad_clip = 5

for (ce_coeff, pg_coeff, vf_coeff) in coeffs:

    for cpt_iter in cpt_iters:

        for (lr_anneal, linear_steps) in lr_anneals:

            for msg_len_ratio in msg_len_ratios:

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

