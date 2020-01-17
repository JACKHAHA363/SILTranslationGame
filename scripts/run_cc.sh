#!/bin/bash
#SBATCH --gres=gpu:p100:1
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --mail-user=luyuchen.paul@gmail.com
#SBATCH --mail-type=END,FAIL


export PYTHONUNBUFFERED=1
exec $@
