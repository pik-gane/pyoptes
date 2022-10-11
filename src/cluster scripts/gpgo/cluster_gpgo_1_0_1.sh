#!/bin/bash

#SBATCH --constraint=broadwell

#SBATCH --qos=short

#SBATCH --job-name=loebkens_gpgo_1_0_1

#SBATCH --account=gane

#SBATCH --output=logs/outputs_gpgo_1_0_1.out

#SBATCH --error=logs/errors_gpgo_1_0_1.err

#SBATCH --workdir=/home/loebkens

#SBATCH --nodes=1      # nodes requested

#SBATCH --ntasks=1      # tasks requested

#SBATCH --cpus-per-task=10

#SBATCH --mem=120000

module load anaconda/5.0.0_py3
source activate bbo
srun -n $SLURM_NTASKS python3 /home/loebkens/pyoptes/src/bb_optimization.py gpgo 20220620_gpgo_rms_nodes_57590_sentinels_3455_budget_4N \
  --path_plot /home/loebkens/pyoptes/data/blackbox_learning/results/ \
  --path_networks /p/projects/ou/labs/gane/optes/mcmc_100nets/data/ \
  --graph syn \
  --prior_mixed_strategies '' \
  --n_nodes 57590 \
  --sentinels 3455 \
  --statistic rms \
  --scale_total_budget 4 \
  --num_cpu_cores 32 \
  --n_runs 10 \
  --max_iterations 10
