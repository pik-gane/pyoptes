#!/bin/bash

#SBATCH --constraint=broadwell

#SBATCH --qos=medium

#SBATCH --job-name=loebkens_np_2_1_0_h_25

#SBATCH --account=gane

#SBATCH --output=logs/outputs_np_2_1_0_h_25.out

#SBATCH --error=logs/errors_np_2_1_0_h_25.err

#SBATCH --workdir=/home/loebkens

#SBATCH --nodes=1      # nodes requested

#SBATCH --ntasks=1      # tasks requested

#SBATCH --cpus-per-task=32

#SBATCH --mem=64000

module load anaconda/5.0.0_py3
source activate bbo
srun -n $SLURM_NTASKS python3 /home/loebkens/pyoptes/src/black-box-optimization.py optimization \
  --optimizer np \
  --name_experiment 20230120_np_mean_nodes_120_h_25 \
  --path_plot /home/loebkens/pyoptes/data/blackbox_learning/results/ \
  --path_networks /home/loebkens/network/data \
  --graph syn \
  --prior_mixed_strategies '' \
  --n_nodes 120 \
  --sentinels 120 \
  --statistic mean \
  --scale_total_budget 1 \
  --r_dim 50 \
  --z_dim 50 \
  --h_dim 25 \
  --num_target 3 \
  --num_context 3 \
  --batch_size 10 \
  --epochs 30 \
  --max_iterations 30 \
  --n_runs 100