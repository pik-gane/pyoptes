#!/bin/bash

#SBATCH --constraint=broadwell

#SBATCH --qos=medium

#SBATCH --job-name=loebkens_gpgo_0_1_0_no_prior_1000_iterations

#SBATCH --account=gane

#SBATCH --output=logs/outputs_gpgo_0_1_0_no_prior_1000_iterations.out

#SBATCH --error=logs/errors_gpgo_0_1_0_no_prior_1000_iterations.err

#SBATCH --workdir=/home/loebkens

#SBATCH --nodes=1      # nodes requested

#SBATCH --ntasks=1      # tasks requested

#SBATCH --cpus-per-task=32

#SBATCH --mem=64000

module load anaconda/5.0.0_py3
source activate bbo
srun -n $SLURM_NTASKS python3 /home/loebkens/pyoptes/src/bb_optimization.py gpgo 20220927_gpgo_mean_nodes_1040_no_prior_1000_iterations \
  --path_plot /home/loebkens/pyoptes/data/blackbox_learning/results/ \
  --path_networks /p/projects/ou/labs/gane/optes/mcmc_100nets/data/ \
  --graph syn \
  --prior_mixed_strategies '' \
  --n_nodes 1040 \
  --sentinels 63 \
  --statistic mean \
  --scale_total_budget 1 \
  --use_prior '' \
  --max_iterations 1000