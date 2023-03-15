'''
Creates scatter plots showing the relationship between the nodes, their attributes and the allocated budget.
'''

from pyoptes import bo_create_graph, bo_scatter_plot, bo_get_node_attributes, bo_choose_sentinels

import numpy as np
from tqdm import tqdm
import os
import glob
import json


def bbo_inspect_test_strategies(path_plot, path_networks):

    paths_experiment_params = glob.glob(os.path.join(path_plot, '**/experiment_hyperparameters.json'), recursive=True)
    for experiment_params in tqdm(paths_experiment_params):

        # get experiment specific hyperparameters
        with open(experiment_params, 'r') as f:
            hyperparameters = json.load(f)

        optimizer = hyperparameters['optimizer_hyperparameters']['optimizer']
        network_type = hyperparameters['simulation_hyperparameters']['graph']
        n_runs = hyperparameters['simulation_hyperparameters']['n_runs']
        n_nodes = hyperparameters['simulation_hyperparameters']['n_nodes']
        sentinels = hyperparameters['simulation_hyperparameters']['sentinels']
        mode_choose_sentinels = hyperparameters['simulation_hyperparameters']['mode_choose_sentinels']

        experiment_directory = os.path.split(experiment_params)[0]
        experiment_name = os.path.split(experiment_directory)[1][9:]
        print(experiment_name)
        all_degrees = []
        all_capacities = []
        all_budgets = []

        for n in range(n_runs):
            # get best strategy
            path_best_strategy = os.path.join(experiment_directory, f'individual/{n}', 'best_parameter.npy')
            best_strategy = np.load(path_best_strategy)

            # get the network and its attribute
            transmissions, capacities, degrees = bo_create_graph(n=n,
                                                                 graph_type=network_type,
                                                                 n_nodes=n_nodes,
                                                                 base_path=path_networks)

            degrees = bo_get_node_attributes(node_attributes=[degrees, None, None],
                                             mode='degree')

            indices_sentinels = bo_choose_sentinels(n_nodes=n_nodes,)

            bo_scatter_plot(path_experiment=os.path.join(experiment_directory, f'individual/{n}'),
                            data_x=degrees,
                            data_y=best_strategy,
                            plot_title=f'Node degree vs allocated budget\n Experiment {experiment_name}',
                            x_label='Node degree',
                            y_label='Budget',
                            plot_name='Scatter-plot_node_degree_vs_budget.png')

            capacities = bo_get_node_attributes(node_attributes=[None, capacities, None],
                                                mode='capacity')

            # scatterplot of the capacity vs the allocated budget
            bo_scatter_plot(path_experiment=os.path.join(experiment_directory, f'individual/{n}'),
                            data_x=capacities,
                            data_y=best_strategy,
                            plot_title=f'Node capacity vs allocated budget\n Experiment {experiment_name}',
                            x_label='Node capacity',
                            y_label='Budget',
                            plot_name='Scatter-plot_node_capacity_vs_budget.png')

    # TODO how do the values in the prior look like, compared to the baseline