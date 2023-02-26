#!/bin/bash

#SBATCH --constraint=broadwell

#SBATCH --qos=short

#SBATCH --job-name=loebkens_gpgo_0_2_0_no_prior

#SBATCH --account=gane

#SBATCH --output=logs/outputs_gpgo_0_2_0_no_prior.out

#SBATCH --error=logs/errors_gpgo_0_2_0_no_prior.err

#SBATCH --workdir=/home/loebkens

#SBATCH --nodes=1      # nodes requested

#SBATCH --ntasks=1      # tasks requested

#SBATCH --cpus-per-task=32

#SBATCH --mem=64000

module load anaconda/5.0.0_py3
source activate bbo
srun -n $SLURM_NTASKS python3 /home/loebkens/pyoptes/src/black-box-optimization.py optimization \
  --optimizer gpgo \
  --name_experiment 20230226_gpgo_95perc_nodes_1040_no_prior \
  --path_plot /home/loebkens/pyoptes/data/blackbox_learning/results/ \
  --path_networks /home/loebkens/network/data \
  --graph syn \
  --prior_mixed_strategies '' \
  --n_nodes 1040 \
  --sentinels 1040 \
  --statistic 95perc \
  --scale_total_budget 1 \
  --use_prior ''