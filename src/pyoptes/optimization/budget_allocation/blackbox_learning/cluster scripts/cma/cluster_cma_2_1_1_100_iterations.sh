#!/bin/bash

#SBATCH --constraint=broadwell

#SBATCH --qos=short

#SBATCH --job-name=loebkens_cma_2_1_1_100_iterations

#SBATCH --account=gane

#SBATCH --output=logs/outputs_cma_2_1_1_100_iterations.out

#SBATCH --error=logs/errors_cma_2_1_1_100_iterations.err

#SBATCH --workdir=/home/loebkens

#SBATCH --nodes=1      # nodes requested

#SBATCH --ntasks=1      # tasks requested

#SBATCH --cpus-per-task=32

#SBATCH --mem=64000

module load anaconda/5.0.0_py3
source activate bbo
srun -n $SLURM_NTASKS python3 /home/loebkens/pyoptes/src/black-box-optimization.py optimization \
  --optimizer cma \
  --name_experiment 20230308_cma_mean_nodes_120_budget_4N_100_iterations \
  --path_plot /home/loebkens/pyoptes/data/blackbox_learning/results/ \
  --path_networks /home/loebkens/network/data \
  --graph syn \
  --n_nodes 120 \
  --sentinels 120 \
  --statistic mean \
  --scale_total_budget 4 \
  --prior_mixed_strategies '' \
  --popsize 9 \
  --n_runs 100 \
  --max_iterations 100