#!/bin/bash

#SBATCH --constraint=broadwell

#SBATCH --qos=medium

#SBATCH --job-name=loebkens_cma_2_2_0

#SBATCH --account=gane

#SBATCH --output=logs/outputs_cma_2_2_0.out

#SBATCH --error=logs/errors_cma_2_2_0.err

#SBATCH --workdir=/home/loebkens

#SBATCH --nodes=1      # nodes requested

#SBATCH --ntasks=1      # tasks requested

#SBATCH --cpus-per-task=32

#SBATCH --mem=64000

module load anaconda/5.0.0_py3
source activate bbo
srun -n $SLURM_NTASKS python3 /home/loebkens/pyoptes/src/black-box-optimization.py optimization \
  --optimizer cma \
  --name_experiment 20230109_cma_95perc_nodes_120 \
  --path_plot /home/loebkens/pyoptes/data/blackbox_learning/results/ \
  --path_networks /p/projects/ou/labs/gane/optes/mcmc_100nets/data/ \
  --graph syn \
  --n_nodes 120 \
  --sentinels 120 \
  --statistic 95perc \
  --scale_total_budget 1 \
  --prior_mixed_strategies '' \
  --popsize 9