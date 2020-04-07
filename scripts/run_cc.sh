#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=20:00:00
#SBATCH --mail-user=luyuchen.paul@gmail.com
#SBATCH --mail-type=END,FAIL


export PYTHONUNBUFFERED=1
exec $@
