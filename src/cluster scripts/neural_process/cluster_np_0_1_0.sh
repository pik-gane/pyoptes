#!/bin/bash

#SBATCH --constraint=broadwell

#SBATCH --qos=short

#SBATCH --job-name=loebkens_np_0_1_0

#SBATCH --account=gane

#SBATCH --output=logs/outputs_np_0_1_0.out

#SBATCH --error=logs/errors_np_0_1_0.err

#SBATCH --workdir=/home/loebkens

#SBATCH --nodes=1      # nodes requested

#SBATCH --ntasks=1      # tasks requested

#SBATCH --cpus-per-task=32

#SBATCH --mem=64000

module load anaconda/5.0.0_py3
source activate bbo
srun -n $SLURM_NTASKS python3 /home/loebkens/pyoptes/src/bb_optimization.py gpgo 20220912_np_mean_nodes_1040 \
  --path_plot /home/loebkens/pyoptes/src/pyoptes/optimization/budget_allocation/blackbox_learning/plots/ \
  --path_networks /p/projects/ou/labs/gane/optes/mcmc_100nets/data/ \
  --graph syn \
  --prior_mixed_strategies '' \
  --n_nodes 1040 \
  --sentinels 1040 \
  --statistic mean \
  --scale_total_budget 1 \
  --use_neural_process True