#!/bin/bash

#SBATCH --constraint=broadwell

#SBATCH --qos=short

#SBATCH --job-name=loebkens_gpgo4

#SBATCH --account=gane

#SBATCH --output=outputs_gpgo4.out

#SBATCH --error=errors_gpgo4.err

#SBATCH --workdir=/home/loebkens

#SBATCH --nodes=1      # nodes requested

#SBATCH --ntasks=1      # tasks requested

#SBATCH --cpus-per-task=32

#SBATCH --mem=64000

module load anaconda/5.0.0_py3
source activate bbo
srun -n $SLURM_NTASKS python3 /home/loebkens/pyoptes/src/bb_optimization.py gpgo 20220513_gpgo_95perc_1040_nodes \
  --path_plot /home/loebkens/pyoptes/src/pyoptes/optimization/budget_allocation/blackbox_learning/plots/ \
  --path_networks /home/loebkens/pyoptes/data \
  --prior_mixed_strategies ''\
  --graph ba \
  --n_nodes 1040 \
  --sentinels 1040 \
  --statistic 95perc \
  --scale_total_budget 1