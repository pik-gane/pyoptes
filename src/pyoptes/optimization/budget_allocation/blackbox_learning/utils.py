import os
import json
import numpy as np
import pylab as plt
from tqdm import tqdm
import pandas as pd
import networkx as nx


def save_results(best_parameter, eval_output, path_experiment):
    """
    Saves the evaluation of the output and the corresponding best parameter.
    @param best_parameter:
    @param eval_output:
    @param path_experiment:
    """
    if not os.path.isdir(path_experiment):
        os.makedirs(path_experiment)

    with open(os.path.join(path_experiment, 'evaluation_output.txt'), 'w') as f:
        f.write(eval_output)

    np.save(os.path.join(path_experiment, 'best_parameter'), best_parameter)


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


def baseline(total_budget, eval_function, n_nodes, parallel, num_cpu_cores):
    """

    @param total_budget:
    @param eval_function:
    @param n_nodes:
    @param parallel:
    @param num_cpu_cores:
    @return:
    """
    x_baseline = np.array([total_budget / n_nodes for _ in range(n_nodes)])

    m, stderr = eval_function(x_baseline,
                              n_simulations=10000,
                              parallel=parallel,
                              num_cpu_cores=num_cpu_cores)

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


def plot_prior(prior, n_simulations, eval_function, parallel, cpu_count, n_runs, path_experiment, n_nodes):
    """

    @param prior:
    @param n_simulations:
    @param eval_function:
    @param parallel:
    @param cpu_count:
    @param n_runs:
    @param path_experiment:

    """
    # TODO move this outside of the function
    y_prior = []
    print(f'Evaluating prior {n_runs} times.')
    for _ in tqdm(range(n_runs), leave=False):

        y_prior.append(evaluate_prior(prior, n_simulations, eval_function, parallel, cpu_count))
    y_prior = np.array(y_prior)

    y_prior_mean = np.mean(y_prior[:, :, 0], axis=0)
    y_prior_stderr = np.mean(y_prior[:, :, 1], axis=0)

    min_y_prior_mean = y_prior_mean.min()
    max_y_prior_mean = y_prior_mean.max()

    plt.bar(range(len(y_prior_mean)), y_prior_mean, label='prior')
    plt.title(f'Objective function evaluation for {len(prior)} strategies')
    plt.xlabel('Prior')
    plt.ylabel('objective function value')
    # TODO move text in the top right corner of the plot
    plt.text(25, 1000, f'min: {min_y_prior_mean:2f}\nmax: {max_y_prior_mean:2f}',
             bbox=dict(facecolor='red', alpha=0.5))
    plt.savefig(os.path.join(path_experiment, f'objective_function_values_prior_{n_nodes}_n_nodes.png'))
    plt.clf()


def create_graphs(n_runs, graph_type):
    """
    Loads n_runs graphs from disk and returns them as a list
    @param n_runs: int, the number of different networks to be loaded
    @param graph_type: string, barabasi-albert or waxman
    @return: list of dictionaries with the graph and the node indices
    """
    assert 0 < n_runs <= 100    # there are only 100 graphs available
    network_list = []
    if graph_type == 'waxman':
        print(f'Loading {n_runs} waxman graphs')
        for n in tqdm(range(n_runs)):
            transmissions_path = os.path.join('../data/', graph_type+'_networks', 'transmissions')
            single_transmission_path = os.path.join(transmissions_path, f'WXtransmission_array_network_{n}.csv')

            transmissions_waxman = pd.read_csv(single_transmission_path)
            waxman = nx.from_pandas_edgelist(transmissions_waxman, source='Source Node', target='Destination Node')
            pos = dict(waxman.nodes.data('pos'))
            static_network = nx.DiGraph(nx.to_numpy_array(waxman))

            network_list.append({'static_network': static_network,
                                 'pos': pos})
    elif graph_type == 'ba':
        print(f'Loading {n_runs} barabasi-albert graphs')
        for n in tqdm(range(n_runs)):
            transmissions_path = os.path.join('../data/', graph_type+'_networks', 'transmissions')
            single_transmission_path = os.path.join(transmissions_path, f'BAtransmission_array_network_{n}.csv')

            transmissions_ba = pd.read_csv(single_transmission_path)
            ba = nx.from_pandas_edgelist(transmissions_ba, source='Source Node', target='Destination Node')
            pos = dict(ba.nodes.data('pos'))
            static_network = nx.DiGraph(nx.to_numpy_array(ba))

            network_list.append({'static_network': static_network,
                                 'pos': pos})
    else:
        Exception(f'Graph type {graph_type} not supported')

    return network_list
