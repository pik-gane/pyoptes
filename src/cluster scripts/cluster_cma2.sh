#!/bin/bash

#SBATCH --constraint=broadwell

#SBATCH --qos=medium

#SBATCH --job-name=loebkens_cma2

#SBATCH --account=gane

#SBATCH --output=logs/outputs_cma2.out

#SBATCH --error=logs/errors_cma2.err

#SBATCH --workdir=/home/loebkens

#SBATCH --nodes=1      # nodes requested

#SBATCH --ntasks=1      # tasks requested

#SBATCH --cpus-per-task=32

#SBATCH --mem=64000

module load anaconda/5.0.0_py3
source activate bbo
srun -n $SLURM_NTASKS python3 /home/loebkens/pyoptes/src/bb_optimization.py cma 20220524_cma_rms_nodes_1040_popsize_9_budget_4N \
  --path_plot /home/loebkens/pyoptes/src/pyoptes/optimization/budget_allocation/blackbox_learning/plots/ \
  --path_networks /p/projects/ou/labs/gane/optes/mcmc_100nets/data/ \
  --graph real \
  --n_nodes 1040 \
  --sentinels 1040 \
  --statistic rms \
  --scale_total_budget 4 \
  --popsize 9

