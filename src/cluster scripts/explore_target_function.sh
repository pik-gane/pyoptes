#!/bin/bash

#SBATCH --constraint=broadwell

#SBATCH --qos=short

#SBATCH --job-name=loebkens_explore

#SBATCH --account=gane

#SBATCH --output=logs/outputs_explore.out

#SBATCH --error=logs/errors_explore.err

#SBATCH --workdir=/home/loebkens

#SBATCH --nodes=1      # nodes requested

#SBATCH --ntasks=1      # tasks requested

#SBATCH --cpus-per-task=32

#SBATCH --mem=64000

module load anaconda/5.0.0_py3
source activate bbo
srun -n $SLURM_NTASKS python3 /home/loebkens/pyoptes/src/black-box-optimization.py explore_target_function \
  --path_plot /home/loebkens/pyoptes/data/blackbox_learning/results/ \
