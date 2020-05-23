#!/usr/bin/env bash
#SBATCH --gres=gpu:titanx:1
#SBATCH --mem=8G
#SBATCH --exclude=eos21,bart14,bart13,leto06,leto23,leto32
#SBATCH --mail-user=luyuchen.paul@gmail.com
#SBATCH --mail-type=FAIL

echo "running on $SLURMD_NODENAME"
export PYTHONUNBUFFERED=1

module load pytorch/1.3.1
source /network/home/luyuchen/py37/bin/activate
exec $@
