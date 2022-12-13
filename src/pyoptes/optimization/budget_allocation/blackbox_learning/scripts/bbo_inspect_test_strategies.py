'''
Creates scatter plots showing the relationship between the nodes, their attributes and the allocated budget.

'''

from pyoptes import create_graph, scatter_plot, get_node_attributes

import numpy as np
import pylab as plt
from tqdm import tqdm
import os
import glob
import json


def inspect_test_strategies(path_plot):

    paths_experiment_params = glob.glob(os.path.join(path_plot, '**/experiment_hyperparameters.json'))
    for experiment_params in tqdm(paths_experiment_params):

        # get experiment specific hyperparameters
        with open(experiment_params, 'r') as f:
            hyperparameters = json.load(f)

        optimizer = hyperparameters['optimizer_hyperparameters']['optimizer']
        network_type = hyperparameters['simulation_hyperparameters']['graph']
        n_runs = hyperparameters['simulation_hyperparameters']['n_runs']
        n_nodes = hyperparameters['simulation_hyperparameters']['n_nodes']
        sentinels = hyperparameters['simulation_hyperparameters']['sentinels']

        experiment_directory = os.path.split(experiment_params)[0]
        experiment_name = os.path.split(experiment_directory)[1][9:]

        if network_type == 'ba' or network_type == 'waxman':
            path_networks = '../data'
        elif network_type == 'syn':
            path_networks = '../networks/data'
        else:
            raise Exception('Network type not supported')

        all_degrees = []
        all_capacities = []
        all_budgets = []

        for n in range(n_runs):
            # get best strategy
            path_best_strategy = os.path.join(experiment_directory, f'individual/{n}', 'best_parameter.npy')
            best_strategy = np.load(path_best_strategy)

            # get the network and its attribute
            transmissions, capacities, degrees = create_graph(n, network_type, n_nodes, path_networks)

            degrees = get_node_attributes([degrees, None, None], 'degree')

            scatter_plot(path_experiment=os.path.join(experiment_directory, f'individual/{n}'),
                         data_x=degrees,
                         data_y=best_strategy,
                         plot_title=f'Node degree vs allocated budget\n Experiment {experiment_name}',
                         x_label='Node degree',
                         y_label='Budget',
                         plot_name='Scatter-plot_node_degree_vs_budget.png'
                         )

            capacities = get_node_attributes([None, capacities, None], 'capacity')

            # scatterplot of the capacity vs the allocated budget
            scatter_plot(path_experiment=os.path.join(experiment_directory, f'individual/{n}'),
                         data_x=capacities,
                         data_y=best_strategy,
                         plot_title=f'Node capacity vs allocated budget\n Experiment {experiment_name}',
                         x_label='Node capacity',
                         y_label='Budget',
                         plot_name='Scatter-plot_node_capacity_vs_budget.png'
                         )

            all_degrees.extend(degrees)
            all_capacities.extend(capacities)
            all_budgets.extend(best_strategy)

        scatter_plot(path_experiment=experiment_directory,
                     data_x=all_degrees,
                     data_y=all_budgets,
                     plot_name='combined_scatter-plot_node_degree_vs_budget.png',
                     plot_title=f'Node degree vs allocated budget over {n_runs} runs.\n Experiment {experiment_name}',
                     x_label='Node degree',
                     y_label='Budget')

        scatter_plot(path_experiment=experiment_directory,
                     data_x=all_capacities,
                     data_y=all_budgets,
                     plot_name='combined_scatter-plot_node_capacity_vs_budget.png',
                     plot_title=f'Node capacity vs allocated budget over {n_runs} runs.\n Experiment {experiment_name}',
                     x_label='Node capacity',
                     y_label='Budget')


    # TODO how to the values in the prior look like, compared to the baseline