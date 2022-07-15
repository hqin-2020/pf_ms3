#!/bin/bash

#SBATCH --job-name=pf_copy
#SBATCH --output=pf_copy.out
#SBATCH --error=pf_copy.err
#SBATCH --time=0-0:25:00
#SBATCH --partition=broadwl
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G

module load python/anaconda-2021.05

python3 copy_final.py  