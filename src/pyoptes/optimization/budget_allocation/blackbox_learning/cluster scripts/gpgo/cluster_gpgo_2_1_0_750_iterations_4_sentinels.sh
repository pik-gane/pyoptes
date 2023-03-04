#!/bin/bash

#SBATCH --constraint=broadwell

#SBATCH --qos=short

#SBATCH --job-name=loebkens_gpgo_2_1_0_750_iterations_4_sentinels

#SBATCH --account=gane

#SBATCH --output=logs/outputs_gpgo_2_1_0_750_iterations_4_sentinels.out

#SBATCH --error=logs/errors_gpgo_2_1_0_750_iterations_4_sentinels.err

#SBATCH --workdir=/home/loebkens

#SBATCH --nodes=1      # nodes requested

#SBATCH --ntasks=1      # tasks requested

#SBATCH --cpus-per-task=32

#SBATCH --mem=64000

module load anaconda/5.0.0_py3
source activate bbo
srun -n $SLURM_NTASKS python3 /home/loebkens/pyoptes/src/black-box-optimization.py optimization \
  --optimizer gpgo \
  --name_experiment 20230305_gpgo_mean_750_iterations_4_sentinels \
  --path_plot /home/loebkens/pyoptes/data/blackbox_learning/results/ \
  --path_networks /home/loebkens/network/data \
  --graph syn \
  --prior_mixed_strategies '' \
  --n_nodes 120 \
  --sentinels 4 \
  --statistic mean \
  --scale_total_budget 1 \
  --max_iterations 750 \
  --n_runs 10
