from pyoptes import choose_high_degree_nodes, create_graphs
from pyoptes.optimization.budget_allocation import target_function as f

import argparse
import numpy as np
import pylab as plt
from tqdm import tqdm
import os
import glob
import json


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # parser.add_argument("name_experiment",
    #                     help="The name of the folder where the results of the optimizer run are saved to.")

    parser.add_argument('--path_plot', default='pyoptes/optimization/budget_allocation/blackbox_learning/plots/',
                        help="Optimizer parameter. Location where all the individual results"
                             " of the optimizers are saved to. "
                             "Default location is 'pyoptes/optimization/budget_allocation/blackbox_learning/plots/'")

    args = parser.parse_args()

    # TODO first, get all experiments that are to be inspected
    paths_experiment_params = glob.glob(os.path.join(args.path_plot, '20220531**/experiment_hyperparameters.json'))
    print(paths_experiment_params)
    # TODO get the experiment parameters from the .json-file
    for experiment_params in paths_experiment_params:

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
        print(experiment_name)

        if network_type == 'ba' or network_type == 'waxman':
            path_networks = '../data'
        elif network_type == 'syn':
            path_networks = '../../networks/data'
        else:
            raise Exception('Network type not supported')
        # load the networks with the experiment specific ..
        network_list = create_graphs(n_runs, network_type, n_nodes, path_networks)

        all_degrees = []
        all_capacities = []
        all_budgets = []

        if optimizer == 'gpgo':

            for n in range(n_runs):
                # get best strategy
                path_best_strategy = os.path.join(experiment_directory, f'individual/{n}', 'best_parameter.npy')
                best_strategy = np.load(path_best_strategy)

                # get the network and its attribute
                network = network_list[n]
                transmissions, capacities, degrees = network

                # get the degrees and capacity of the network, sorted by node indice
                if network_type == 'syn':
                    degrees = []
                else:
                    # degrees_sorted = sorted(degrees, key=lambda degrees: degrees[1], reverse=True)
                    degrees = [i[1] for i in degrees]

                all_degrees.extend(degrees)
                all_capacities.extend(capacities)
                all_budgets.extend(best_strategy)

            plt.clf()
            plt.scatter(all_degrees, all_budgets)
            plt.title(f'Node degree vs allocated budget over {n_runs} runs.\n Experiment {experiment_name}')
            plt.xlabel('Node degree')
            plt.ylabel('Budget')

            plt.savefig(os.path.join(experiment_directory, 'Scatter-plot_node_degree_vs_budget.png'))
            # plt.show()

        elif optimizer == 'cma' and n_nodes == 120:

            for n in range(n_runs):
                # get best strategy
                path_best_strategy = os.path.join(experiment_directory, f'individual/{n}', 'best_parameter.npy')
                best_strategy = np.load(path_best_strategy)

                # get the network and its attribute
                network = network_list[n]
                transmissions, capacities, degrees = network

                # get the degrees and capacity of the network, sorted by node indice
                if network_type == 'syn':
                    degrees = []
                else:
                    # degrees_sorted = sorted(degrees, key=lambda degrees: degrees[1], reverse=True)
                    degrees = [i[1] for i in degrees]

                all_degrees.extend(degrees)
                all_capacities.extend(capacities)
                all_budgets.extend(best_strategy)

        elif optimizer == 'cma' and n_nodes == 1040:

            for n in range(n_runs):
                # get best strategy
                path_best_strategy = os.path.join(experiment_directory, f'individual/{n}', 'best_parameter.npy')
                xbest = glob.glob(os.path.join(experiment_directory, f'individual/{n}', 'budget/*.npy'))
                for x in xbest:
                    b = np.load(x)
                    # print(b)
                    # print('mean, min, max', b.mean(), b.min(), b.max())
                    # print(np.exp(b))
                    # print(sum(np.exp(b)))
                    b = b - np.max(b)
                    b = 1040 * np.exp(b) / sum(np.exp(b))
                    print('mean, min, max', b.mean(), b.min(), b.max())
                    print('---')

                best_strategy = np.load(path_best_strategy)
                print(best_strategy)
                print('mean, min, max', best_strategy.mean(), best_strategy.min(), best_strategy.max(), '\n')

                # get the network and its attribute
                network = network_list[n]
                transmissions, capacities, degrees = network

                # get the degrees and capacity of the network, sorted by node indice
                if network_type == 'syn':
                    degrees = []
                else:
                    # degrees_sorted = sorted(degrees, key=lambda degrees: degrees[1], reverse=True)
                    degrees = [i[1] for i in degrees]

                all_degrees.extend(degrees)
                all_capacities.extend(capacities)
                all_budgets.extend(best_strategy)

            all_degrees.extend(degrees)
            all_capacities.extend(capacities)
            all_budgets.extend(best_strategy)

            plt.clf()
            plt.scatter(all_degrees, all_budgets)
            plt.title(f'Node degree vs allocated budget over {n_runs} runs.\n Experiment {experiment_name}')
            plt.xlabel('Node degree')
            plt.ylabel('Budget')

            plt.savefig(os.path.join(experiment_directory, 'Scatter-plot_node_degree_vs_budget.png'))
            # plt.show()

    # TODO how to the values in the prior look like, compared to the baseline