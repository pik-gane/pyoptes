#!/bin/bash

#SBATCH --constraint=broadwell

#SBATCH --qos=short

#SBATCH --job-name=loebkens_np_1_1_1_50_iterations

#SBATCH --account=gane

#SBATCH --output=logs/outputs_np_1_1_1_50_iterations.out

#SBATCH --error=logs/errors_np_1_1_1_50_iterations.err

#SBATCH --workdir=/home/loebkens

#SBATCH --nodes=1      # nodes requested

#SBATCH --ntasks=1      # tasks requested

#SBATCH --cpus-per-task=32

#SBATCH --mem=64000

module load anaconda/5.0.0_py3
source activate bbo
srun -n $SLURM_NTASKS python3 /home/loebkens/pyoptes/src/black-box-optimization.py optimization \
  --optimizer np \
  --name_experiment 20230308_np_mean_nodes_57590_4N_budget_50_iterations \
  --path_plot /home/loebkens/pyoptes/data/blackbox_learning/results/ \
  --path_networks /home/loebkens/network/data \
  --graph syn \
  --prior_mixed_strategies '' \
  --n_nodes 57590 \
  --sentinels 1329 \
  --statistic mean \
  --scale_total_budget 4 \
  --r_dim 50 \
  --z_dim 50 \
  --h_dim 50 \
  --num_target 3 \
  --num_context 3 \
  --batch_size 10 \
  --epochs 30 \
  --max_iterations 50 \
  --n_runs 10