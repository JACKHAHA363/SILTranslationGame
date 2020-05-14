#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=20:00:00
#SBATCH --account=def-bengioy

export PYTHONUNBUFFERED=1
exec $@
