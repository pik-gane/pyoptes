#!/bin/bash

#SBATCH --constraint=broadwell

#SBATCH --qos=medium

#SBATCH --job-name=loebkens_gpgo_2_1_0_sent_12_1000_iterations_part1

#SBATCH --account=gane

#SBATCH --output=logs/outputs_gpgo_2_1_0_sent_12_1000_iterations_part1.out

#SBATCH --error=logs/errors_gpgo_2_1_0_sent_12_1000_iterations_part1.err

#SBATCH --workdir=/home/loebkens

#SBATCH --nodes=1      # nodes requested

#SBATCH --ntasks=1      # tasks requested

#SBATCH --cpus-per-task=32

#SBATCH --mem=64000

module load anaconda/5.0.0_py3
source activate bbo
srun -n $SLURM_NTASKS python3 /home/loebkens/pyoptes/src/bb_optimization.py gpgo 20221024_gpgo_mean_sent_12_1000_max_iterations \
  --path_plot /home/loebkens/pyoptes/data/blackbox_learning/results/ \
  --path_networks /home/loebkens/network/data \
  --graph syn \
  --prior_mixed_strategies '' \
  --n_nodes 120 \
  --sentinels 12 \
  --statistic mean \
  --scale_total_budget 1 \
  --max_iterations 1000 \
  --n_runs 100 \
  --n_runs 50 \
  --n_runs_start 0
