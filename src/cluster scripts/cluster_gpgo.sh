#!/bin/bash

#SBATCH --constraint=broadwell

#SBATCH --qos=short

#SBATCH --job-name=loebkens_gpgo

#SBATCH --account=gane

#SBATCH --output=logs/outputs_gpgo.out

#SBATCH --error=logs/errors_gpgo.err

#SBATCH --workdir=/home/loebkens

#SBATCH --nodes=1      # nodes requested

#SBATCH --ntasks=1      # tasks requested

#SBATCH --cpus-per-task=10

#SBATCH --mem=120000

module load anaconda/5.0.0_py3
source activate bbo
srun -n $SLURM_NTASKS python3 /home/loebkens/pyoptes/src/bb_optimization.py gpgo 20220529_gpgo_95perc_nodes_57590 \
  --path_plot /home/loebkens/pyoptes/src/pyoptes/optimization/budget_allocation/blackbox_learning/plots/ \
  --path_networks /p/projects/ou/labs/gane/optes/mcmc_100nets/data/ \
  --graph syn \
  --prior_mixed_strategies '' \
  --n_nodes 57590 \
  --sentinels 57590 \
  --statistic 95perc \
  --scale_total_budget 1 \
  --num_cpu_cores 1 \
  --n_runs 10
