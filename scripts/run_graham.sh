#!/bin/bash
#SBATCH --gres=gpu:p100:1
#SBATCH --mem=16G
#SBATCH --time=4:00:00

export PYTHONUNBUFFERED=1
exec $@
