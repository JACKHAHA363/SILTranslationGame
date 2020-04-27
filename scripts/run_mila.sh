#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --exclude=eos21,bart14
#SBATCH --mail-user=luyuchen.paul@gmail.com
#SBATCH --mail-type=END,FAIL

echo "running on $SLURMD_NODENAME"
export PYTHONUNBUFFERED=1

#source /network/home/luyuchen/miniconda2/etc/profile.d/conda.sh
#conda activate py36
source /network/home/luyuchen/py36/bin/activate
exec $@
