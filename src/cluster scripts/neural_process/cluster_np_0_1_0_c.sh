#!/bin/bash

#SBATCH --constraint=broadwell

#SBATCH --qos=short

#SBATCH --job-name=loebkens_np_0_1_0_big_np

#SBATCH --account=gane

#SBATCH --output=logs/outputs_np_0_1_0_big_np.out

#SBATCH --error=logs/errors_np_0_1_0_big_np.err

#SBATCH --workdir=/home/loebkens

#SBATCH --nodes=1      # nodes requested

#SBATCH --ntasks=1      # tasks requested

#SBATCH --cpus-per-task=32

#SBATCH --mem=64000

module load anaconda/5.0.0_py3
source activate bbo
srun -n $SLURM_NTASKS python3 /home/loebkens/pyoptes/src/bb_optimization.py np 20220915_big_np_np_mean_nodes_1040 \
  --path_plot /home/loebkens/pyoptes/src/pyoptes/optimization/budget_allocation/blackbox_learning/plots/ \
  --path_networks /p/projects/ou/labs/gane/optes/mcmc_100nets/data/ \
  --graph syn \
  --prior_mixed_strategies '' \
  --n_nodes 1040 \
  --sentinels 1040 \
  --statistic mean \
  --scale_total_budget 1 \
  --y_dim 1 \
  --r_dim 100 \
  --z_dim 100 \
  --h_dim 100 \
  --num_target 3 \
  --num_context 3 \
  --batch_size 10 \
  --epochs 100