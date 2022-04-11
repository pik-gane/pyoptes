import os
import json
import numpy as np
import pylab as plt
from tqdm import tqdm
import pandas as pd
import networkx as nx
from time import time


def save_results(best_test_strategy, path_experiment, output, save_test_strategy=True):
    """
    Saves the evaluation of the output and the corresponding best parameter.
    @param save_test_strategy:
    @param output:
    @param best_test_strategy: numpy array, the best parameter the optimizer has found
    @param path_experiment:
    """
    if not os.path.isdir(path_experiment):
        os.makedirs(path_experiment)

    with open(os.path.join(path_experiment, 'evaluation_output.txt'), 'w') as f:
        f.write(output)

    if save_test_strategy:
        np.save(os.path.join(path_experiment, 'best_parameter'), best_test_strategy)


def save_hyperparameters(hyperparameters, base_path):
    """
    Saves the parameters for a training run of a model,
    Parameters are saved as a json file.
    Args:
        hyperparameters: dict,
        base_path: string, base path where the results are saved to
    """
    if not os.path.isdir(base_path):
        os.makedirs(base_path)

    json_path = os.path.join(base_path, 'experiment_hyperparameters.json')

    with open(json_path, 'w') as fp:
        json.dump(hyperparameters, fp, sort_keys=True, indent=4)
    print('saved hyperparameters\n')


def choose_high_degree_nodes(node_degrees, n):
    """
    Returns the indices of n nodes with the highest degrees.
    @param node_degrees: list, contains indices of nodes and their degree
    @param n: int, number of nodes to be returned
    @return: list of ints
    """
    # sort list of nodes by degree
    nodes_sorted = sorted(node_degrees, key=lambda node_degrees: node_degrees[1], reverse=True)
    # save the indices of the n highest degree nodes
    indices_highest_degree_nodes = [i[0] for i in nodes_sorted[:n]]
    return indices_highest_degree_nodes


def map_low_dim_x_to_high_dim(x, n_nodes, node_indices):
    """
    Map the values in an array x to an empty array of size n_nodes. The location of each xi is based on node_indices.
    @param x: numpy array, output array to be extended to size n_nodes
    @param n_nodes: int, defines size of output array
    @param node_indices: list of integers
    @return: numpy array of size n_node, with values of x at indices node_indices
    """
    # create a dummy vector to be filled with the values of x at the appropriate indices
    x_true = np.zeros(n_nodes)
    for i, xi in zip(node_indices, x):
        x_true[i] = xi
    return x_true


# TODO plot priors and standard errors
def create_test_strategy_prior(n_nodes, node_degrees, node_capacities, total_budget, sentinels):
    """
    Creates a list of test strategies to be used as a prior.
    First element in the list is a strategy where the budget is uniformly distributed over all sentinel nodes
    that the objective function is using.
    @param sentinels:
    @param n_nodes: int, number of nodes in SI-simulation graph
    @param node_degrees: list, contains indices of nodes and their degree
    @param node_capacities: list,capacity of each node
    @param total_budget: float, total budget for the allocation
    @return: list numpy arrays, each array contains the values of a test strategy
    """
    # print('Assembling prior from test strategies created by different heuristics.\n')
    test_strategy_prior = []

    # specify increasing number of sentinels
    sentinels_list = [int(n_nodes / 6), int(n_nodes / 12), int(n_nodes / 24)]

    # sort list of nodes by degree
    nodes_degrees_sorted = sorted(node_degrees, key=lambda node_degrees: node_degrees[1], reverse=True)

    # sort nodes by capacities
    nodes_capacities = [(i, c) for i, c in
                        enumerate(node_capacities)]  # node_contains only capacities, add node_index
    nodes_capacities_sorted = sorted(nodes_capacities, key=lambda nodes_capacities: nodes_capacities[1],
                                     reverse=True)

    test_strategy_parameter = 'number\tdescription'

    # create strategy for nodes = sentinels used with the objective function
    indices_highest_degree_nodes = [i[0] for i in nodes_degrees_sorted[:sentinels]]
    x_sentinels = np.array([total_budget / sentinels for _ in range(sentinels)])
    test_strategy_prior.append(map_low_dim_x_to_high_dim(x_sentinels, n_nodes, indices_highest_degree_nodes))

    test_strategy_parameter += f'\n0\tuniform distribution over all {n_nodes} nodes'

    n = 1
    for i, s in enumerate(sentinels_list):
        # ------
        # create strategy for s highest degree nodes, budget is allocated uniformly
        indices_highest_degree_nodes = [i[0] for i in nodes_degrees_sorted[:s]]
        x_sentinels = np.array([total_budget / s for _ in range(s)])
        test_strategy_prior.append(map_low_dim_x_to_high_dim(x_sentinels, n_nodes, indices_highest_degree_nodes))

        test_strategy_parameter += f'\n{n}\tuniform distribution over {s} highest degree nodes'
        n += 1
        # ------
        # create strategy for s highest capacity nodes, budget is allocated uniformly
        indices_highest_capacity_nodes = [i[0] for i in nodes_capacities_sorted[:s]]
        x_sentinels = np.array([total_budget / s for _ in range(s)])
        test_strategy_prior.append(map_low_dim_x_to_high_dim(x_sentinels, n_nodes, indices_highest_capacity_nodes))

        test_strategy_parameter += f'\n{n}\tuniform distribution over {s} highest capacity nodes'
        n += 1
        # ------
        # create strategies that are a mix of the highest degree and highest capacity nodes
        for k in range(s)[1:]:

            # get the highest degree nodes and highest capacity nodes
            indices_highest_degree_nodes = [i[0] for i in nodes_degrees_sorted[:k]]
            indices_highest_capacity_nodes = [i[0] for i in nodes_capacities_sorted[:s-k]]
            # check whether node indices would appear twice and remove the duplicates
            # TODO maybe there is a better method for this check??
            indices_combined = list(set(indices_highest_degree_nodes) | set(indices_highest_capacity_nodes))
            # because of the missing nodes the strategies might violate the sum constraint (lightly)
            # therefore of this the allocated budget is smaller or greater than the total budget
            x_sentinels = np.array([total_budget / len(indices_combined) for _ in indices_combined])
            test_strategy_prior.append(map_low_dim_x_to_high_dim(x_sentinels, n_nodes, indices_combined))

            test_strategy_parameter += f'\n{n}\tuniform distribution over ' \
                                       f'{k} highest degree nodes and {s-k} highest capacity nodes'
            n += 1

    return test_strategy_prior, test_strategy_parameter


def baseline(total_budget, eval_function, n_nodes, parallel, num_cpu_cores, statistic):
    """

    @param total_budget:
    @param eval_function:
    @param n_nodes:
    @param parallel:
    @param num_cpu_cores:
    @return:
    """
    # distribute budget uniformly over all nodes
    x_baseline = np.array([total_budget / n_nodes for _ in range(n_nodes)])

    m, stderr = eval_function(x_baseline,
                              n_simulations=10000,
                              parallel=parallel,
                              num_cpu_cores=num_cpu_cores,
                              statistic=statistic
                              )

    return m, stderr


def test_function(x, *args, **kwargs):
    """
    Quadratic function just for test purposes
    @param x: numpy array, input vector
    @return: float, square of first element of input vector
    """
    return x[0]**2


def evaluate_prior(prior, n_simulations, eval_function, parallel, num_cpu_cores):
    """
    Evaluate the strategies in the prior and return the mean and standard error
    @param prior: list of numpy arrays, each array contains the values of a test strategy
    @param n_simulations: int, number of simulations to be performed
    @param eval_function: function, function to be evaluated
    @param parallel: bool, whether to use parallelization or not
    @param num_cpu_cores: int, number of cpu cores to be used for parallelization
    @return: A list of the mean and standard error for every strategy in the prior
    """
    y_prior = []
    # print(f'Evaluating prior with {n_simulations} simulations')
    for strategy in tqdm(prior, leave=False):
        m, stderr = eval_function(strategy,
                                  n_simulations=n_simulations,
                                  parallel=parallel,
                                  num_cpu_cores=num_cpu_cores)

        y_prior.append(np.array([m, stderr]))

    return np.array(y_prior)


def create_graphs(n_runs, graph_type, n_nodes):
    """
    Loads n_runs graphs from disk and returns them as a list
    @param n_runs: int, the number of different networks to be loaded
    @param graph_type: string, barabasi-albert or waxman
    @return: list of dictionaries with the graph and the node indices
    """
    assert 0 < n_runs <= 100    # there are only 100 graphs available
    network_list = []
    network_path = os.path.join('../data/', graph_type + '_networks', f'{n_nodes}')
    if graph_type == 'waxman':
        print(f'Loading {n_runs} waxman graphs')
        for n in tqdm(range(n_runs)):
            transmission_path = os.path.join(network_path, f'WX{n}', 'transmissions.txt')
            transmissions_waxman = pd.read_csv(transmission_path, header=None).to_numpy()

            capacities_path = os.path.join(network_path, f'WX{n}', 'capacity.txt')
            capacities_waxman = pd.read_csv(capacities_path, header=None).to_numpy().squeeze()

            degrees_path = os.path.join(network_path, f'WX{n}', 'degree.txt')
            degrees_waxman = pd.read_csv(degrees_path, header=None).to_numpy()

            network_list.append([transmissions_waxman,
                                 np.int_(capacities_waxman),
                                 degrees_waxman])
    elif graph_type == 'ba':
        print(f'Loading {n_runs} barabasi-albert graphs')
        for n in tqdm(range(n_runs)):
            single_transmission_path = os.path.join(network_path, f'BA{n}', 'transmissions.txt')
            transmissions_ba = pd.read_csv(single_transmission_path, header=None).to_numpy()

            capacities_path = os.path.join(network_path, f'BA{n}', 'capacity.txt')
            capacities_ba = pd.read_csv(capacities_path, header=None).to_numpy().squeeze()

            degrees_path = os.path.join(network_path, f'BA{n}', 'degree.txt')
            degrees_ba = pd.read_csv(degrees_path, header=None).to_numpy()

            network_list.append([transmissions_ba,
                                 np.int_(capacities_ba),
                                 degrees_ba])
    else:
        Exception(f'Graph type {graph_type} not supported')

    return network_list
