#!/bin/bash

#SBATCH --constraint=broadwell

#SBATCH --qos=medium

#SBATCH --job-name=loebkens_cma

#SBATCH --account=gane

#SBATCH --output=outputs_cma.out

#SBATCH --error=errors_cma.err

#SBATCH --workdir=/home/loebkens

#SBATCH --nodes=1      # nodes requested

#SBATCH --ntasks=1      # tasks requested

#SBATCH --cpus-per-task=32

#SBATCH --mem=64000

module load anaconda/5.0.0_py3
source activate bbo
srun -n $SLURM_NTASKS python3 /home/loebkens/pyoptes/src/bb_optimization.py cma 20220508_cma_1040_nodes_95perc \
  --path_plot /home/loebkens/pyoptes/src/pyoptes/optimization/budget_allocation/blackbox_learning/plots/ \
  --path_networks /home/loebkens/pyoptes/data \
  --graph ba \
  --n_nodes 1040 \
  --sentinels 1040 \
  --statistic 95perc \
  --scale_total_budget 1