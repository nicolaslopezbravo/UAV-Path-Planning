#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -p gpu
module load tensorflow
python test.py
