#!/bin/bash

#SBATCH --constraint=broadwell

#SBATCH --qos=medium

#SBATCH --job-name=loebkens_cma6

#SBATCH --account=gane

#SBATCH --output=outputs_cma6.out

#SBATCH --error=errors_cma6.err

#SBATCH --workdir=/home/loebkens

#SBATCH --nodes=1      # nodes requested

#SBATCH --ntasks=1      # tasks requested

#SBATCH --cpus-per-task=32

#SBATCH --mem=64000

module load anaconda/5.0.0_py3
source activate bbo
srun -n $SLURM_NTASKS python3 /home/loebkens/pyoptes/src/bb_optimization.py cma 20220513_cma_rms_popsize_2 \
  --path_plot /home/loebkens/pyoptes/src/pyoptes/optimization/budget_allocation/blackbox_learning/plots/ \
  --path_networks /home/loebkens/pyoptes/data \
  --graph ba \
  --n_nodes 120 \
  --sentinels 120 \
  --statistic rms \
  --scale_total_budget 1 \
  --popsize 2

