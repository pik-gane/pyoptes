#!/bin/bash

#SBATCH --constraint=broadwell

#SBATCH --qos=medium

#SBATCH --job-name=loebkens_cma_1_0_0_part2

#SBATCH --account=gane

#SBATCH --output=logs/outputs_cma_1_0_0_part2.out

#SBATCH --error=logs/errors_cma_1_0_0_part2.err

#SBATCH --workdir=/home/loebkens

#SBATCH --nodes=1      # nodes requested

#SBATCH --ntasks=1      # tasks requested

#SBATCH --cpus-per-task=32

#SBATCH --mem=120000

module load anaconda/5.0.0_py3
source activate bbo
srun -n $SLURM_NTASKS python3 /home/loebkens/pyoptes/src/black-box-optimization.py optimization \
  --optimizer cma \
  --name_experiment 20230226_cma_rms_nodes_57590_sentinels_1329 \
  --path_plot /home/loebkens/pyoptes/data/blackbox_learning/results/ \
  --path_networks /home/loebkens/network/data \
  --graph syn \
  --n_nodes 57590 \
  --sentinels 1329 \
  --statistic rms \
  --scale_total_budget 1 \
  --popsize 9 \
  --num_cpu_cores 32 \
  --prior_mixed_strategies '' \
  --n_runs 10 \
  --n_runs 5 \
  --n_runs_start 5

