#!/usr/bin/env bash

export PYTHONUNBUFFERED=1

source /network/home/luyuchen/miniconda2/etc/profile.d/conda.sh
conda activate py36
exec $@
