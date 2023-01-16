#!/bin/bash

#SBATCH --constraint=broadwell

#SBATCH --qos=medium

#SBATCH --job-name=loebkens_cma_0_1_1_sigma_1

#SBATCH --account=gane

#SBATCH --output=logs/outputs_cma_0_1_1_sigma_1.out

#SBATCH --error=logs/errors_cma_0_1_1_sigma_1.err

#SBATCH --workdir=/home/loebkens

#SBATCH --nodes=1      # nodes requested

#SBATCH --ntasks=1      # tasks requested

#SBATCH --cpus-per-task=32

#SBATCH --mem=64000

module load anaconda/5.0.0_py3
source activate bbo
srun -n $SLURM_NTASKS python3 /home/loebkens/pyoptes/src/black-box-optimization.py optimization \
  --optimizer cma \
  --name-experiment 20230109_cma_mean_nodes_1040_budget_4N_sigma_1\
  --path_plot /home/loebkens/pyoptes/data/blackbox_learning/results/ \
  --path_networks /home/loebkens/network/data \
  --graph syn \
  --n_nodes 1040 \
  --sentinels 1040 \
  --statistic mean \
  --scale_total_budget 4 \
  --prior_mixed_strategies '' \
  --popsize 9 \
  --scale_sigma 1

