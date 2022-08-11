'''
Compute the target function value for different number of sentinels (either highest degree, transmissions, capacities)

'''

from pyoptes.optimization.budget_allocation import target_function as f
from pyoptes import create_graph, compute_average_otf_and_stderr

import argparse
import numpy as np
from tqdm import tqdm
from scipy.stats.mstats import mjci

# TODO look into maltes file in the repo
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--n_runs', type=int, default=100,
                        help='')
    parser.add_argument("--n_simulations", type=int, default=10000,
                        help="Si-simulation parameter. Sets the number of runs the for the SI-model. "
                             "Higher values of n_simulations lower the variance of the output of the simulation. "
                             "Default value is 1000.")

    parser.add_argument('--graph_type', choices=['waxman', 'ba', 'syn'], default='syn',
                        help='Si-simulation parameter. Set the type of graph the simulation uses.'
                             ' Either Waxman or Barabasi-Albert (ba) can be used. Default is Barabasi-Albert.')

    parser.add_argument('--path_networks', default='../../networks/data',
                        help='Location where the networks are saved to. '
                             'Path on cluster. /p/projects/ou/labs/gane/optes/mcmc_100nets/data'
                             '/p/projects/ou/labs/gane/optes/mcmc_100nets/data/')
    args = parser.parse_args()

    #
    nodes = [120, 1040, 57590]
