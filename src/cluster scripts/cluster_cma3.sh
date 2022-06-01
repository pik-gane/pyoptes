#!/bin/bash

#SBATCH --constraint=broadwell

#SBATCH --qos=medium

#SBATCH --job-name=loebkens_cma3

#SBATCH --account=gane

#SBATCH --output=logs/outputs_cma3.out

#SBATCH --error=logs/errors_cma3.err

#SBATCH --workdir=/home/loebkens

#SBATCH --nodes=1      # nodes requested

#SBATCH --ntasks=1      # tasks requested

#SBATCH --cpus-per-task=32

#SBATCH --mem=120000

module load anaconda/5.0.0_py3
source activate bbo
srun -n $SLURM_NTASKS python3 /home/loebkens/pyoptes/src/bb_optimization.py cma 20220524_cma_rms_nodes_57590_popsize_9 \
  --path_plot /home/loebkens/pyoptes/src/pyoptes/optimization/budget_allocation/blackbox_learning/plots/ \
  --path_networks /p/projects/ou/labs/gane/optes/mcmc_100nets/data/ \
  --graph syn \
  --n_nodes 57590 \
  --sentinels 57590 \
  --statistic rms \
  --scale_total_budget 1 \
  --popsize 9 \
  --num_cpu_cores 10 \
  --prior_mixed_strategies '' \
  --n_runs 10

