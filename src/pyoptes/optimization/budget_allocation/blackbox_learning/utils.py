import os
import json
import numpy as np
from tqdm import tqdm
import pandas as pd
import networkx as nx


def softmax(x):
    """
    Softmax function.
    @param x: budget vector, numpy array
    @return: scaled budget vector, numpy array
    """
    x = x - np.max(x)  # for numerical stability. Prevents infinities in the denominator.
    x = np.exp(x) / sum(np.exp(x))
    return x


def compute_average_otf_and_stderr(list_otf, list_stderr, n_runs):
    average_best_otf = np.mean(list_otf, axis=0)
    s = np.mean(list_stderr, axis=0)
    v = np.var(list_otf, axis=0)
    average_best_otf_stderr = np.sqrt(v/n_runs + s**2)

    return average_best_otf, average_best_otf_stderr


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


# TODO add function to load raw data from file
def save_raw_data(list_best_otf, list_best_otf_stderr, list_baseline_otf, list_baseline_otf_stderr,
                  list_ratio_otf, list_best_solution_history, list_stderr_history, list_time_for_optimization,
                  list_all_prior_tf, list_all_prior_stderr,
                  path_experiment):
    """
    Saves the raw data of the optimization process.
    @param list_best_otf:
    @param list_best_otf_stderr:
    @param list_baseline_otf:
    @param list_baseline_otf_stderr:
    @param list_ratio_otf:
    @param list_best_solution_history:
    @param list_stderr_history:
    @param list_time_for_optimization:
    @param path_experiment:
    """
    path_experiment = os.path.join(path_experiment, 'raw_data')
    if not os.path.isdir(path_experiment):
        os.makedirs(path_experiment)
    # turn lists into numpy arrays and save them in path_experiment
    np.save(os.path.join(path_experiment, 'list_best_otf'), np.array(list_best_otf))
    np.save(os.path.join(path_experiment, 'list_best_otf_stderr'), np.array(list_best_otf_stderr))
    np.save(os.path.join(path_experiment, 'list_baseline_otf'), np.array(list_baseline_otf))
    np.save(os.path.join(path_experiment, 'list_baseline_otf_stderr'), np.array(list_baseline_otf_stderr))
    np.save(os.path.join(path_experiment, 'list_ratio_otf'), np.array(list_ratio_otf))
    np.save(os.path.join(path_experiment, 'list_best_solution_history'), np.array(list_best_solution_history))
    np.save(os.path.join(path_experiment, 'list_stderr_history'), np.array(list_stderr_history))
    np.save(os.path.join(path_experiment, 'list_time_for_optimization'), np.array(list_time_for_optimization))
    np.save(os.path.join(path_experiment, 'list_all_prior_tf'), list_all_prior_tf)
    np.save(os.path.join(path_experiment, 'list_all_prior_stderr'), list_all_prior_stderr)


def load_raw_data(path_experiment):
    """

    @param path_experiment:
    @return:
    """
    list_best_otf = np.load(os.path.join(path_experiment, 'list_best_otf.npy'))
    list_best_otf_stderr = np.load(os.path.join(path_experiment, 'list_best_otf_stderr.npy'))
    list_baseline_otf = np.load(os.path.join(path_experiment, 'list_baseline_otf.npy'))
    list_baseline_otf_stderr = np.load(os.path.join(path_experiment, 'list_baseline_otf_stderr.npy'))
    list_ratio_otf = np.load(os.path.join(path_experiment, 'list_ratio_otf.npy'))
    list_best_solution_history = np.load(os.path.join(path_experiment, 'list_best_solution_history.npy'))
    list_stderr_history = np.load(os.path.join(path_experiment, 'list_stderr_history.npy'))
    list_time_for_optimization = np.load(os.path.join(path_experiment, 'list_time_for_optimization.npy'))
    list_all_prior_tf = np.load(os.path.join(path_experiment, 'list_all_prior_tf.npy'))
    list_all_prior_stderr = np.load(os.path.join(path_experiment, 'list_all_prior_stderr.npy'), )
    return {'list_best_otf': list_best_otf, 'list_best_otf_stderr': list_best_otf_stderr,
            'list_baseline_otf': list_baseline_otf, 'list_baseline_otf_stderr': list_baseline_otf_stderr,
            'list_ratio_otf': list_ratio_otf,
            'list_best_solution_history': list_best_solution_history, 'list_stderr_history': list_stderr_history,
            'list_time_for_optimization': list_time_for_optimization,
            'list_all_prior_tf': list_all_prior_tf, 'list_all_prior_stderr': list_all_prior_stderr}


# TODO change function to "choose_n_sentinels", allowing switching between n highest degrees and capacities
# maybe even a combination of both
def choose_high_degree_nodes(node_degrees, n_nodes, sentinels):
    """
    Returns the indices with the highest degrees. The nodes are sorted by degree, starting with the highest degree node.

    @param n_nodes: int, number of total nodes in the network
    @param node_degrees: list, contains indices of nodes and their degree
    @param sentinels: int, number of nodes to be returned
    @return: list of ints
    """
    # the synthetic networks only gives one a sorted list of node indices of the highest degree nodes
    # excluding some nodes (slaughter houses). To work with the prior the missing nodes have to be added
    if len(node_degrees) < n_nodes:
        # # create a list of all n_nodes node indices
        # all_node_indices = list(range(n_nodes))
        # # get the indices of the nodes that are missing in node_degrees
        # missing_nodes = list(set(all_node_indices) - set(node_degrees))
        # add the missing node indices to the end of the node_degrees list
        # TODO this is useless, degrees are converted into indices
        nodes_degrees_sorted = sorted(node_degrees, reverse=True)
        missing_nodes = np.zeros(n_nodes-len(node_degrees))
        indices_highest_degree_nodes = [*nodes_degrees_sorted, *missing_nodes]
    else:
        # sort list of nodes by degree and get their indices
        nodes_degrees_sorted = sorted(node_degrees, key=lambda node_degrees: node_degrees[1], reverse=True)
        indices_highest_degree_nodes = [i[0] for i in nodes_degrees_sorted]

    return indices_highest_degree_nodes[:sentinels]


def map_low_dim_x_to_high_dim(x, n_nodes, node_indices):
    """
    Map the values in an array x to an empty array of size n_nodes. The location of each xi is based on node_indices.
    @param x: numpy array, output array to be extended to size n_nodes
    @param n_nodes: int, defines size of output array
    @param node_indices: list of integers
    @return: numpy array of size n_node, with values of x at indices node_indices
    """
    # print('x, node_indices: ', np.shape(x), np.shape(node_indices))
    assert np.shape(x) == np.shape(node_indices)
    # create a dummy vector to be filled with the values of x at the appropriate indices
    x_true = np.zeros(n_nodes)
    for i, xi in zip(node_indices, x):
        x_true[i] = xi
    return x_true


# TODO restrict prior to a fixed number of strategies
# TODO make prior with sentinels < n_nodes work with gpgo
# To make the prior work with an objective function where the number of sentinels is lower than the number of nodes
# the budget had to always be allocated between the sentinels. This already disallows the creation of the baseline.
#
def create_test_strategy_prior(n_nodes, node_degrees, node_capacities, total_budget,
                               sentinels, mixed_strategies=True, only_baseline=False):
    """
    Creates a list of test strategies to be used as a prior.
    First element in the list is a strategy where the budget is uniformly distributed over all sentinel nodes
    that the objective function is using.
    @param only_baseline: bool, sets whether only the baseline strategy is used as a prior
    @param mixed_strategies: bool, sets whether strategies with a mix of high degree and high capacity nodes are used
    @param sentinels:
    @param n_nodes: int, number of nodes in the SI-simulation graph
    @param node_degrees: list, contains indices of nodes and their degree
    @param node_capacities: list, capacity of each node
    @param total_budget: float, total budget for the allocation
    @return: three lists,
        1. list of numpy arrays, each array defining the budget distribution over a number of sentinels
        2. list of node indices for the sentinels in each strategy
        3. a string containing descriptions for each test strategy
    """
    prior_test_strategies = []
    prior_node_indices = []
    test_strategy_parameter = 'number\tdescription'     # string containing descriptions for each test strategy

    # specify increasing number of sentinels TODO why these numbers in specific ?
    sentinels_list = [int(sentinels / 6), int(sentinels / 12), int(sentinels / 24)]

    # get the (sorted) indices of the highest degree nodes
    indices_highest_degree_nodes = choose_high_degree_nodes(node_degrees, n_nodes, sentinels)

    # node_capacities contains only capacities, add node_index sort nodes by capacities and get their indices
    nodes_capacities = [(i, c) for i, c in enumerate(node_capacities)]
    nodes_capacities_sorted = sorted(nodes_capacities, key=lambda nodes_capacities: nodes_capacities[1],
                                     reverse=True)
    indices_highest_capacity_nodes = [i[0] for i in nodes_capacities_sorted]

    # ------------------------------------------------------------------------
    # TODO has to be fixed to distribute over n_nodes instead of sentinels
    # the baseline strategy (equally distributed budget over all sentinels)
    x_sentinels = np.array([total_budget / sentinels for _ in range(sentinels)])
    prior_test_strategies.append(map_low_dim_x_to_high_dim(x_sentinels, n_nodes,
                                                           indices_highest_degree_nodes[:sentinels]))
    prior_node_indices.append(indices_highest_degree_nodes)

    test_strategy_parameter += f'\n0\tuniform distribution over all {n_nodes} nodes'

    if not only_baseline:
        n = 1   # index for test strategies
        for i, s in enumerate(sentinels_list):
            # ------
            # create strategy for s highest degree nodes, budget is allocated uniformly
            x_sentinels = np.array([total_budget / s for _ in range(s)])
            prior_test_strategies.append(map_low_dim_x_to_high_dim(x_sentinels, n_nodes,
                                                                   indices_highest_degree_nodes[:s]))
            prior_node_indices.append(indices_highest_degree_nodes)

            test_strategy_parameter += f'\n{n}\tuniform distribution over {s} highest degree nodes'
            n += 1
            # ------
            # create strategy for s highest capacity nodes, budget is allocated uniformly
            x_sentinels = np.array([total_budget / s for _ in range(s)])
            prior_test_strategies.append(map_low_dim_x_to_high_dim(x_sentinels, n_nodes,
                                                                   indices_highest_capacity_nodes[:s]))
            prior_node_indices.append(indices_highest_capacity_nodes)

            test_strategy_parameter += f'\n{n}\tuniform distribution over {s} highest capacity nodes'
            n += 1
            # ------
            # create strategies that are a mix of the highest degree and highest capacity nodes
            if mixed_strategies:
                for k in range(s)[1:]:

                    # get the highest degree nodes and highest capacity nodes and
                    # check whether node indices would appear twice and remove the duplicates
                    # TODO maybe there is a better method for this check??
                    indices_combined = list(set(indices_highest_degree_nodes[:k]) |
                                            set(indices_highest_capacity_nodes[:s-k]))
                    # because of the missing nodes the strategies might violate the sum constraint (lightly)
                    # therefore of this the allocated budget is smaller or greater than the total budget
                    x_sentinels = np.array([total_budget / len(indices_combined) for _ in indices_combined])
                    prior_test_strategies.append(map_low_dim_x_to_high_dim(x_sentinels, n_nodes, indices_combined))
                    prior_node_indices.append(indices_combined)

                    test_strategy_parameter += f'\n{n}\tuniform distribution over ' \
                                               f'{k} highest degree nodes and {s-k} highest capacity nodes'
                    n += 1

    return prior_test_strategies, prior_node_indices, test_strategy_parameter


def baseline(total_budget, eval_function, n_nodes, parallel, num_cpu_cores, statistic):
    """
    Creates a test strategy where the total budget is uniformly allocated to all nodes.
    Evaluates the test strategy with the given evaluation function for 10000 simulation runs.
    @param statistic: function, aggregates the results of the simulation runs
    @param total_budget: int, total budget for the test strategy
    @param eval_function: SI-simulation
    @param n_nodes: int, number of nodes in the network
    @param parallel: bool, whether to use parallelization for the SI-simulation
    @param num_cpu_cores: int, number of cpu cores to use for parallelization
    @return: mean and standard error of uniform baseline and the corresponding uniform test strategy
    """
    # distribute budget uniformly over all nodes
    x_baseline = np.array([total_budget / n_nodes for _ in range(n_nodes)])

    m, stderr = eval_function(x_baseline,
                              n_simulations=10000,
                              parallel=parallel,
                              num_cpu_cores=num_cpu_cores,
                              statistic=statistic
                              )

    return m, stderr, x_baseline


def test_function(x, *args, **kwargs):
    """
    Quadratic function just for test purposes
    @param x: numpy array, input vector
    @return: float, square of first element of input vector
    """
    return x[0]**2


def evaluate_prior(prior, n_simulations, eval_function, parallel, num_cpu_cores, statistic):
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
    for strategy in tqdm(prior, leave=False):
        m, stderr = eval_function(strategy,
                                  n_simulations=n_simulations,
                                  parallel=parallel,
                                  num_cpu_cores=num_cpu_cores,
                                  statistic=statistic)

        y_prior.append(np.array([m, stderr]))

    return np.array(y_prior)


def create_graphs(n_runs, graph_type, n_nodes, base_path='../data/'):
    """
    Loads n_runs graphs from disk and returns them as a list
    @param n_nodes:
    @param base_path:
    @param n_runs: int, the number of different networks to be loaded
    @param graph_type: string, barabasi-albert or waxman
    @return: list of dictionaries with the graph and the node indices
    """
    assert 0 < n_runs <= 100    # there are only 100 graphs available
    network_list = []
    if graph_type == 'waxman':
        network_path = os.path.join(base_path, graph_type + '_networks', f'{n_nodes}')
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
        network_path = os.path.join(base_path, graph_type + '_networks', f'{n_nodes}')
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
    elif graph_type == 'syn':
        network_path = os.path.join(base_path, f'Synset{n_nodes}-180')
        print(f'Loading {n_runs} synthetic graphs')
        for n in tqdm(range(n_runs)):
            transmissions_path = os.path.join(network_path, f'syndata{n}', 'dataset.txt')
            transmissions = pd.read_csv(transmissions_path, header=None)
            transmissions = transmissions[[2, 2, 0, 1, 3]]
            transmissions = transmissions.to_numpy()

            capacities_path = os.path.join(network_path, f'syndata{n}', 'barn_size.txt')
            capacities = pd.read_csv(capacities_path, header=None)
            capacities = capacities.iloc[0][:n_nodes].to_numpy()

            degrees_path = os.path.join(network_path, f'syndata{n}', 'degreen.txt')
            degrees = pd.read_csv(degrees_path, header=None)
            degrees = degrees.iloc[0][:-1].to_numpy(dtype=np.int64)

            if len(degrees) < n_nodes:
                # synthetic networks
                missing_nodes = np.zeros(n_nodes - len(degrees))
                degree_nodes = [*degrees, *missing_nodes]
                degrees = [[i, d] for i, d in enumerate(degree_nodes)]

            network_list.append([transmissions,
                                 capacities,
                                 degrees])

    else:
        Exception(f'Graph type {graph_type} not supported')

    return network_list
