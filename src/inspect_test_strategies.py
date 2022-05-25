from pyoptes import save_hyperparameters, save_results, plot_prior, create_graphs
from pyoptes.optimization.budget_allocation import target_function as f

import argparse
import numpy as np
from tqdm import tqdm
import os


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("name_experiment",
                        help="The name of the folder where the results of the optimizer run are saved to.")

    parser.add_argument("--n_nodes", type=int, default=120, choices=[120, 1040, 57590],
                        help="Si-simulation parameter. "
                             "Defines the number of nodes used by the SI-model to create a graph. "
                             "Default value is 120 nodes.")
    parser.add_argument('--n_runs', type=int, default=100,
                        help='The number of times the optimizer is run. Results are then averaged over all runs.'
                             'Default is 100 runs.')
    parser.add_argument('--path_plot', default='pyoptes/optimization/budget_allocation/blackbox_learning/plots/',
                        help="Optimizer parameter. Location where all the individual results"
                             " of the optimizers are saved to. "
                             "Default location is 'pyoptes/optimization/budget_allocation/blackbox_learning/plots/'")
    parser.add_argument('--path_networks', default='../data',
                        help='Location where the networks are saved to. '
                             'Path on cluster. /p/projects/ou/labs/gane/optes/mcmc_100nets/data'
                             '/p/projects/ou/labs/gane/optes/mcmc_100nets/data/')

    parser.add_argument('--graph_type', choices=['waxman', 'ba', 'syn'], default='ba',
                        help='Si-simulation parameter. Set the type of graph the simulation uses.'
                             ' Either Waxman or Barabasi-Albert (ba) can be used. Default is Barabasi-Albert.')

    parser.add_argument('--delta_t_symptoms', type=int, default=60,
                        help='Si-simulation parameter.. Sets the time (in days) after which an infection is detected'
                             ' automatically. Default is 60 days')
    parser.add_argument('--p_infection_by_transmission', type=float, default=0.5,
                        help='Si-simulation parameter. The probability of how likely a trade animal '
                             'infects other animals. Default is 0.5.')
    parser.add_argument('--expected_time_of_first_infection', type=int, default=30,
                        help='Si-simulation parameter. The expected time (in days) after which the first infection occurs. ')

    args = parser.parse_args()

    network_list = create_graphs(args.n_runs, args.graph_type, args.n_nodes, args.path_networks)

    for n, network in enumerate(network_list[:args.n_runs]):

        # unpack the properties of the network
        transmissions, capacities, degrees = network

        #
        path_best_strategy = os.path.join(args.path_plot, args.name_experiment, f'raw/{n}', 'best_parameter.npy')
        best_strategy = np.load(path_best_strategy)
        print('shape of best_strategy: ', best_strategy.shape)