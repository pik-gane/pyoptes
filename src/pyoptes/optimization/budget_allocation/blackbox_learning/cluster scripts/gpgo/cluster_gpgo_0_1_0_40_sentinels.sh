#!/bin/bash

#SBATCH --constraint=broadwell

#SBATCH --qos=short

#SBATCH --job-name=loebkens_gpgo_0_1_0_40_sentinels

#SBATCH --account=gane

#SBATCH --output=logs/outputs_gpgo_0_1_0_40_sentinels.out

#SBATCH --error=logs/errors_gpgo_0_1_0_40_sentinels.err

#SBATCH --workdir=/home/loebkens

#SBATCH --nodes=1      # nodes requested

#SBATCH --ntasks=1      # tasks requested

#SBATCH --cpus-per-task=32

#SBATCH --mem=64000

module load anaconda/5.0.0_py3
source activate bbo
srun -n $SLURM_NTASKS python3 /home/loebkens/pyoptes/src/black-box-optimization.py optimization \
  --optimizer gpgo \
  --name_experiment 20230307_gpgo_mean_nodes_1040_40_sentinels \
  --path_plot /home/loebkens/pyoptes/data/blackbox_learning/results/ \
  --path_networks /home/loebkens/network/data \
  --graph syn \
  --prior_mixed_strategies '' \
  --n_nodes 1040 \
  --sentinels 40 \
  --statistic mean \
  --scale_total_budget 1