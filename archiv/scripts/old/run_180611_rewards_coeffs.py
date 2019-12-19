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

jobname = "180611_rewards_coeffs"
mem = 16
hours = 24
slurm_dir = home_dir + "180611_rewards_coeffs" if not interactive else ""
if not interactive and not os.path.exists(slurm_dir):
    os.mkdir(slurm_dir)
event_path = "event_180611_rewards_coeffs"
model_path = "model_180611_rewards_coeffs"
log_path = "log_180611_rewards_coeffs"
decoding_path = "decoding_180611_rewards_coeffs"

cpt_iters = [500, 5000]

nll_rewards = ["mean", "sum"]

coeffs = [ (1.0, 1.0, 0.01), # 2
           (1.0, 10.0, 0.01),
           (1.0, 100.0, 0.01),
           (1.0, 1.0, 0.1),
           (1.0, 10.0, 0.1),
           (1.0, 100.0, 0.1),
         ]

msg_len_ratio = 1.0

lrs = [1e-3, 3e-4, 1e-4] # 2

lr1 = ["half", "linear"] # 2
lr2 = [0, 100000]
lr_anneals = [(a,b) for (a,b) in zip(lr1, lr2)]

drop_ratio = 0.3
eval_every = 200
save_every = -1
max_steps = 300000
job_idx = 0
grad_clip = 5.0

for (ce_coeff, pg_coeff, vf_coeff) in coeffs: # 6
    for (lr_anneal, linear_steps) in lr_anneals: # 2
        for lr in lrs: # 3
            for cpt_iter in cpt_iters: # 2
                for nll_reward in nll_rewards: # 2

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
                    subprocess.call(cmd, shell=True)

                    #sys.exit()
                    time.sleep(0.001)

print ("Submitted {} jobs".format(job_idx))
