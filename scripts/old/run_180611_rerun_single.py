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

jobname = "180611_rerun_single"
mem = 24
hours = 12
slurm_dir = home_dir + "180611_rerun_single" if not interactive else ""
if not interactive and not os.path.exists(slurm_dir):
    os.mkdir(slurm_dir)
event_path = "event_180611_rerun_single"
model_path = "model_180611_rerun_single"
log_path = "log_180611_rerun_single"
decoding_path = "decoding_180611_rerun_single"

# python /private/home/jasonleeinf/scratch/groundcomms/src/run.py --load_vocab --load_dataset --setup single --pair fr_en --model RNN --eval_every 500 --save_every 500 --drop_ratio 0.4 --lr_anneal linear --linear_anneal_steps 30000 --max_training_steps 50000 --event_path event_180601 --model_path model_180601 --early_stop

pairs = ["fr_en", "en_de"]
drop_ratios = [0.2, 0.3, 0.4, 0.5]
lrs = [1e-3, 3e-4, 1e-4] # 3
save_every = 100
eval_every = 100

lr1 = ["half", "linear"] # 2
lr2 = [0, 10000]
lr_anneals = [(a,b) for (a,b) in zip(lr1, lr2)]
max_steps = 30000

job_idx = 0
for pair in pairs: # 2
    for drop_ratio in drop_ratios: # 4
        for lr in lrs: # 3 
            for (lr_anneal, linear_steps) in lr_anneals: # 3
                job_idx += 1

                options = "sbatch -J {}_{} --gres=gpu:1 --mem {}GB --time={}:00:00 --output={}/%j.out".format(jobname, job_idx, mem, hours, slurm_dir)

                cmd = "python {} \
                        --load_vocab --load_dataset \
                        --setup single\
                        --pair {}\
                        --model RNNAttn\
                        --eval_every {}\
                        --save_every {}\
                        --drop_ratio {}\
                        --lr {}\
                        --lr_anneal {}\
                        --linear_anneal_steps {}\
                        --max_training_steps {}\
                        --event_path {} \
                        --log_path {} \
                        --model_path {}".format(run, pair, eval_every, save_every, drop_ratio, lr, lr_anneal, linear_steps, max_steps, event_path, log_path, model_path)

                if not interactive:
                    cmd += " --no_tqdm"

                cmd = re.sub( '\s+', ' ', cmd ).strip()
                #print (cmd, "\n")

                cmd = '{} {} \"{}\"'.format(options, slurm_file, cmd)
                #print (cmd, "\n")
                subprocess.call(cmd, shell=True)
                time.sleep(0.01)

print ("Submitted {} jobs".format(job_idx))
