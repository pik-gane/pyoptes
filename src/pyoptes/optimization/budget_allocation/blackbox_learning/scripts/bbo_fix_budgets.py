'''
Fixes saved budgets if they violate the constraints and have the wrong dimensions
'''

from pyoptes import bo_create_graph, bo_scatter_plot, bo_get_node_attributes
from pyoptes import bo_softmax, bo_map_low_dim_x_to_high_dim, bo_choose_sentinels

import numpy as np
from tqdm import tqdm
import os
import glob
import json


def bbo_fix_budgets(path_plot, path_networks):

    paths_experiment_params = glob.glob(os.path.join(path_plot, '**/experiment_hyperparameters.json'), recursive=True)
    for experiment_params in tqdm(paths_experiment_params):

        # get experiment specific hyperparameters
        with open(experiment_params, 'r') as f:
            hyperparameters = json.load(f)

        network_type = hyperparameters['simulation_hyperparameters']['graph']
        n_runs = hyperparameters['simulation_hyperparameters']['n_runs']
        n_nodes = hyperparameters['simulation_hyperparameters']['n_nodes']
        sentinels = hyperparameters['simulation_hyperparameters']['sentinels']
        total_budget = hyperparameters['simulation_hyperparameters']['total_budget']

        experiment_directory = os.path.split(experiment_params)[0]
        experiment_name = os.path.split(experiment_directory)[1][9:]

        for n in range(n_runs):
            # get best strategy
            path_best_strategy = os.path.join(experiment_directory, f'individual/{n}', 'best_parameter.npy')
            best_strategy = np.load(path_best_strategy)

            # check whether np.sum of best_strategy is lower or equal to total_budget
            if np.sum(best_strategy) > total_budget:
                # scale the budget with a softmax
                best_strategy = total_budget * bo_softmax(best_strategy)

                # for the mapping to the correct number of nodes info about the network is needed
                # get the network and its attribute
                transmissions, capacities, degrees = bo_create_graph(n, network_type, n_nodes, path_networks)

                node_attributes = [degrees, capacities, transmissions]
                node_indices = bo_choose_sentinels(node_attributes=node_attributes,
                                                   sentinels=sentinels,
                                                   mode='degree')

                best_strategy = bo_map_low_dim_x_to_high_dim(x=best_strategy,
                                                             number_of_nodes=n_nodes,
                                                             node_indices=node_indices)

                np.save(path_best_strategy, best_strategy)
            else:
                print(f'Budget for {experiment_name} run {n} is correct')

        print('np.shape of degrees, capacities, transmissions', np.shape(degrees), np.shape(capacities),
              np.shape(transmissions))

        print('best strategy shape', best_strategy.shape)
        print(experiment_directory, '\n')