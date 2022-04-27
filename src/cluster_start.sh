#!/bin/bash

#SBATCH --constraint=broadwell

#SBATCH --qos=medium

#SBATCH --job-name=loebkens

#SBATCH --account=gane

#SBATCH --output=outputs.out

#SBATCH --error=errors.err

#SBATCH --workdir=/home/loebkens

#SBATCH --nodes=1      # nodes requested

#SBATCH --ntasks=1      # tasks requested

#SBATCH --cpus-per-task=32

#SBATCH --mem=64000

module load anaconda/2020.07
source activate pik
srun -n $SLURM_NTASKS python3 bb_optimization.py cma 2022_cma_100_runs