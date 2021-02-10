#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --mem=2GB
#SBATCH --job-name=1-nmr

module purge

singularity exec --nv \
	    --overlay /scratch/aqd215/pytorch1.7.0-cuda11.0.ext3:ro \
	    /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif \
	    bash -c "source /ext3/env.sh; python ../nmr_shift_data/1-3-nmr.py"
