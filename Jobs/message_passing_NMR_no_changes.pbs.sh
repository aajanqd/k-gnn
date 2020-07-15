#!/bin/bash
#
#SBATCH --job-name=message_passing_NMR_no_changes
#SBATCH --nodes=1 
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=20:00:00
#SBATCH --mem=20GB
module purge
module load anaconda3/5.3.1
module load gcc/6.3.0
module load cuda/10.1.105

source ~/py3.7/bin/activate
python ../nmr_mpnn-master/run_train.py