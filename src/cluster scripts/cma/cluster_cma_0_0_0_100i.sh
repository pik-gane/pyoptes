#!/bin/bash

#SBATCH --constraint=broadwell

#SBATCH --qos=medium

#SBATCH --job-name=loebkens_cma_0_0_0_100i

#SBATCH --account=gane

#SBATCH --output=logs/outputs_cma_0_0_0_100i.out

#SBATCH --error=logs/errors_cma_0_0_0_100i.err

#SBATCH --workdir=/home/loebkens

#SBATCH --nodes=1      # nodes requested

#SBATCH --ntasks=1      # tasks requested

#SBATCH --cpus-per-task=32

#SBATCH --mem=64000

module load anaconda/5.0.0_py3
source activate bbo
srun -n $SLURM_NTASKS python3 /home/loebkens/pyoptes/src/bb_optimization.py cma 20220613_cma_rms_nodes_1040_100i \
  --path_plot /home/loebkens/pyoptes/src/pyoptes/optimization/budget_allocation/blackbox_learning/plots/ \
  --path_networks /p/projects/ou/labs/gane/optes/mcmc_100nets/data/ \
  --graph syn \
  --n_nodes 1040 \
  --sentinels 1040 \
  --statistic rms \
  --scale_total_budget 1 \
  --prior_mixed_strategies '' \
  --popsize 9 \
  --max_iterations 100

