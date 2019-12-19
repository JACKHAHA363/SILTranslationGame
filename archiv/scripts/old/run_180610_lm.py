import sys
import time
import os
import re
import subprocess
import pprint

interactive = False

slurm_file = "/private/home/jasonleeinf/dotfiles/run/run_learnfair.sh"
run = "/private/home/jasonleeinf/scratch/groundcomms/src/lm/main.py"
home_dir = "/private/home/jasonleeinf/slurm/"

jobname = "180610_lm"
mem = 16
hours = 24
slurm_dir = home_dir + "180610_lm" if not interactive else ""
if not interactive and not os.path.exists(slurm_dir):
    os.mkdir(slurm_dir)
event_path = "event_180610_lm"
model_path = "model_180610_lm"
log_path = "log_180610_lm"
decoding_path = "decoding_180610_lm"

# python main.py --cuda --emsize 650 --nhid 650 --dropout 0.5 --epochs 40 --tied
datas = ["wikitext2", "coco", "multi30k"]
arches = ["Feedforward"]
#arches = ["RNN", "Feedforward"]
#filter_widths = {"RNN":[0],
#                 "Feedforward":[2,4,6]}
filter_widths = {"RNN":[0],
                 "Feedforward":[8,10]}
lrs = [20, 10, 5]
dropouts = [0.5, 0.3]

job_idx = 0
for data in datas:
    for arch in arches:
        for filter_width in filter_widths[arch]:
            for lr in lrs:
                for dropout in dropouts:
                    job_idx += 1

                    options = "sbatch -J {}_{} --gres=gpu:1 --mem {}GB --time={}:00:00 --output={}/%j.out".format(jobname, job_idx, mem, hours, slurm_dir)

                    cmd = "python {} \
                            --data {}\
                            --arch {}\
                            --filter_width {}\
                            --lr {}\
                            --dropout {}\
                            --cuda\
                            --emsize 650\
                            --nhid 650\
                            --epochs 40\
                            --tied".format(run, data, arch, filter_width, lr, dropout)

                    cmd = re.sub( '\s+', ' ', cmd ).strip()
                    #print (cmd, "\n")

                    cmd = '{} {} \"{}\"'.format(options, slurm_file, cmd)
                    cmd = "cd /private/home/jasonleeinf/scratch/groundcomms/src/lm; " + cmd
                    subprocess.call(cmd, shell=True)

                    #sys.exit()
                    time.sleep(0.001)

print (job_idx)
