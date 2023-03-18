#!/bin/bash

#SBATCH --constraint=broadwell

#SBATCH --qos=short

#SBATCH --job-name=loebkens_cma_0_1_2_100_iterations_part3

#SBATCH --account=gane

#SBATCH --output=logs/outputs_cma_0_1_2_100_iterations_part3.out

#SBATCH --error=logs/errors_cma_0_1_2_100_iterations_part3.err

#SBATCH --workdir=/home/loebkens

#SBATCH --nodes=1      # nodes requested

#SBATCH --ntasks=1      # tasks requested

#SBATCH --cpus-per-task=32

#SBATCH --mem=64000

module load anaconda/5.0.0_py3
source activate bbo
srun -n $SLURM_NTASKS python3 /home/loebkens/pyoptes/src/black-box-optimization.py optimization \
  --optimizer cma \
  --name_experiment 20230308_cma_mean_nodes_1040_budget_12N_100_iterations\
  --path_plot /home/loebkens/pyoptes/data/blackbox_learning/results/ \
  --path_networks /home/loebkens/network/data \
  --graph syn \
  --n_nodes 1040 \
  --sentinels 1040 \
  --statistic mean \
  --scale_total_budget 12 \
  --prior_mixed_strategies '' \
  --popsize 9 \
  --n_runs 34 \
  --n_runs_start 66 \
  --max_iterations 100

