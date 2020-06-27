#!/bin/bash
#
#SBATCH --job-name=NMR_paper_no_changes
#SBATCH --nodes=1 
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --mem=20GB
module purge
module load anaconda3/5.3.1
module load gcc/6.3.0
module load cuda/10.1.105

source ~/pyenv/py3.7/bin/activate
conda install -c conda-forge rdkit