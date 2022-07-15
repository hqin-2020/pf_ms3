#!/bin/bash

#SBATCH --job-name=pf_ms3
#SBATCH --output=pf_ms3.%A_%a.out
#SBATCH --error=pf_ms3.%A_%a.err
#SBATCH --time=0-2:00:00
#SBATCH --partition=broadwl
#SBATCH --nodes=1
#SBATCH --cpus-per-task=28
#SBATCH --mem-per-cpu=2000
#SBATCH --array=1-130
#SBATCH --constraint=edr

module load python/anaconda-2021.05

srun python3 Run_no_para_update_$SLURM_ARRAY_TASK_ID.py  