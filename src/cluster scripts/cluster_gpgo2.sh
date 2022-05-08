#!/bin/bash

#SBATCH --constraint=broadwell

#SBATCH --qos=short

#SBATCH --job-name=loebkens_gpgo2

#SBATCH --account=gane

#SBATCH --output=outputs_gpgo2.out

#SBATCH --error=errors_gpgo2.err

#SBATCH --workdir=/home/loebkens

#SBATCH --nodes=1      # nodes requested

#SBATCH --ntasks=1      # tasks requested

#SBATCH --cpus-per-task=32

#SBATCH --mem=64000

module load anaconda/5.0.0_py3
source activate bbo
srun -n $SLURM_NTASKS python3 /home/loebkens/pyoptes/src/bb_optimization.py gpgo 20220508_gpgo_mean_12N_budget \
  --path_plot /home/loebkens/pyoptes/src/pyoptes/optimization/budget_allocation/blackbox_learning/plots/ \
  --path_networks /home/loebkens/pyoptes/data \
  --prior_mixed_strategies True\
  --graph ba \
  --n_nodes 120 \
  --sentinels 120 \
  --statistic mean \
  --scale_total_budget 12