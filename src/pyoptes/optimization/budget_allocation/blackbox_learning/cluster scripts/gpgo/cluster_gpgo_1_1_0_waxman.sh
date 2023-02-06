#!/bin/bash

#SBATCH --constraint=broadwell

#SBATCH --qos=short

#SBATCH --job-name=loebkens_gpgo_1_1_0_waxman

#SBATCH --account=gane

#SBATCH --output=logs/outputs_gpgo_1_1_0_waxman.out

#SBATCH --error=logs/errors_gpgo_1_1_0_waxman.err

#SBATCH --workdir=/home/loebkens

#SBATCH --nodes=1      # nodes requested

#SBATCH --ntasks=1      # tasks requested

#SBATCH --cpus-per-task=10

#SBATCH --mem=120000

module load anaconda/5.0.0_py3
source activate bbo
srun -n $SLURM_NTASKS python3 /home/loebkens/pyoptes/src/black-box-optimization.py optimization \
  --optimizer gpgo \
  --name_experiment 20230206_gpgo_mean_nodes_57590_sentinels_3455_waxman \
  --path_plot /home/loebkens/pyoptes/data/blackbox_learning/results/ \
  --path_networks /home/loebkens/network/data \
  --graph syn \
  --prior_mixed_strategies '' \
  --n_nodes 57590 \
  --sentinels 3455 \
  --statistic mean \
  --scale_total_budget 1 \
  --num_cpu_cores 32 \
  --n_runs 10 \
  --max_iterations 10 \
  --graph_type waxman

