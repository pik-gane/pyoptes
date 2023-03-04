#!/bin/bash

#SBATCH --constraint=broadwell

#SBATCH --qos=medium

#SBATCH --job-name=loebkens_cma_2_1_0_1000_iterations_4_sentinels

#SBATCH --account=gane

#SBATCH --output=logs/outputs_cma_2_1_0_1000_iterations_4_sentinels.out

#SBATCH --error=logs/errors_cma_2_1_0_1000_iterations_4_sentinels.err

#SBATCH --workdir=/home/loebkens

#SBATCH --nodes=1      # nodes requested

#SBATCH --ntasks=1      # tasks requested

#SBATCH --cpus-per-task=32

#SBATCH --mem=64000

module load anaconda/5.0.0_py3
source activate bbo
srun -n $SLURM_NTASKS python3 /home/loebkens/pyoptes/src/black-box-optimization.py optimization \
  --optimizer cma \
  --name_experiment 20230305_cma_mean_nodes_1040_1000_iterations_40_sentinels \
  --path_plot /home/loebkens/pyoptes/data/blackbox_learning/results/ \
  --path_networks /home/loebkens/network/data \
  --graph syn \
  --n_nodes 120 \
  --sentinels 4 \
  --statistic mean \
  --scale_total_budget 1 \
  --prior_mixed_strategies '' \
  --popsize 9 \
  --max_iterations 1000 \
  --n_runs 10