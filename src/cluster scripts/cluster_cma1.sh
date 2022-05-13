#!/bin/bash

#SBATCH --constraint=broadwell

#SBATCH --qos=medium

#SBATCH --job-name=loebkens_cma1

#SBATCH --account=gane

#SBATCH --output=outputs_cma1.out

#SBATCH --error=errors_cma1.err

#SBATCH --workdir=/home/loebkens

#SBATCH --nodes=1      # nodes requested

#SBATCH --ntasks=1      # tasks requested

#SBATCH --cpus-per-task=32

#SBATCH --mem=64000

module load anaconda/5.0.0_py3
source activate bbo
srun -n $SLURM_NTASKS python3 /home/loebkens/pyoptes/src/bb_optimization.py cma 20220513_cma_nodes_1040_mean \
  --path_plot /home/loebkens/pyoptes/src/pyoptes/optimization/budget_allocation/blackbox_learning/plots/ \
  --path_networks /home/loebkens/pyoptes/data \
  --graph ba \
  --n_nodes 1040 \
  --sentinels 1040 \
  --statistic mean \
  --scale_total_budget 1

