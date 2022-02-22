#!/bin/bash
#SBATCH --partition=gpu-volta
#SBATCH --cpus-per-task=10
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --time=300:00:00
#SBATCH --chdir=.
#SBATCH --output=../../bash-log/microglia-cell-density-c-v-5_%j.txt
#SBATCH --error=../../bash-log/microglia-cell-density-c-v-5_%j.txt
#SBATCH --job-name=microglia-cell-density-c-v-5

module load CUDA/11.4
python density_mask_train_unet.py