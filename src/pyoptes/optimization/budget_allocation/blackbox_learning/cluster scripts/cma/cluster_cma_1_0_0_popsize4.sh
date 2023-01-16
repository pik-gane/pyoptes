#!/bin/bash

#SBATCH --constraint=broadwell

#SBATCH --qos=medium

#SBATCH --job-name=loebkens_cma_1_0_0_popsize4

#SBATCH --account=gane

#SBATCH --output=logs/outputs_cma_1_0_0_popsize4.out

#SBATCH --error=logs/errors_cma_1_0_0_popsize4.err

#SBATCH --workdir=/home/loebkens

#SBATCH --nodes=1      # nodes requested

#SBATCH --ntasks=1      # tasks requested

#SBATCH --cpus-per-task=32

#SBATCH --mem=120000

module load anaconda/5.0.0_py3
source activate bbo
srun -n $SLURM_NTASKS python3 /home/loebkens/pyoptes/src/black-box-optimization.py optimization \
  --optimizer cma \
  --name_experiment 20230109_cma_nodes_57590_sentinels_3455_popsize_4 \
  --path_plot /home/loebkens/pyoptes/data/blackbox_learning/results/ \
  --path_networks /home/loebkens/network/data \
  --graph syn \
  --n_nodes 57590 \
  --sentinels 3455 \
  --statistic rms \
  --scale_total_budget 1 \
  --popsize 4 \
  --num_cpu_cores 32 \
  --prior_mixed_strategies '' \
  --n_runs 10

