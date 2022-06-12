#!/bin/bash

#SBATCH --constraint=broadwell

#SBATCH --qos=short

#SBATCH --job-name=loebkens_gpgo_0_0_2_only_baseline

#SBATCH --account=gane

#SBATCH --output=logs/outputs_gpgo_0_0_2_only_baseline.out

#SBATCH --error=logs/errors_gpgo_0_0_2_only_baseline.err

#SBATCH --workdir=/home/loebkens

#SBATCH --nodes=1      # nodes requested

#SBATCH --ntasks=1      # tasks requested

#SBATCH --cpus-per-task=32

#SBATCH --mem=64000

module load anaconda/5.0.0_py3
source activate bbo
srun -n $SLURM_NTASKS python3 /home/loebkens/pyoptes/src/bb_optimization.py gpgo 20220612_gpgo_rms_nodes_1040_budget_12N_only_baseline \
  --path_plot /home/loebkens/pyoptes/src/pyoptes/optimization/budget_allocation/blackbox_learning/plots/ \
  --path_networks /p/projects/ou/labs/gane/optes/mcmc_100nets/data/ \
  --graph syn \
  --prior_mixed_strategies '' \
  --prior_only_baseline True \
  --n_nodes 1040 \
  --sentinels 1040 \
  --statistic rms \
  --scale_total_budget 12