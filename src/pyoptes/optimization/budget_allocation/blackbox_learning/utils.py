import os
import json
import numpy as np


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
def create_test_strategy_prior(n_nodes, node_degrees, node_capacities, total_budget):
    """
    Creates a list of test strategies to be used as a prior.
    @param n_nodes: int, number of nodes in SI-simulation graph
    @param node_degrees: list, contains indices of nodes and their degree
    @param node_capacities: list,capacity of each node
    @param total_budget: float, total budget for the allocation
    @return:
    """
    test_strategy_prior = []

    # specify increasing number of sentinels
    sentinels = [int(n_nodes / 6), int(n_nodes / 12), int(n_nodes / 24)]

    # sort list of nodes by degree
    nodes_degrees_sorted = sorted(node_degrees, key=lambda node_degrees: node_degrees[1], reverse=True)

    # sort nodes by capacities
    nodes_capacities = [(i, c) for i, c in
                        enumerate(node_capacities)]  # node_contains only capacities, add node_index
    nodes_capacities_sorted = sorted(nodes_capacities, key=lambda nodes_capacities: nodes_capacities[1],
                                     reverse=True)

    for s in sentinels:
        # ------
        # create strategy for s highest degree nodes, budget is allocated uniformly
        indices_highest_degree_nodes = [i[0] for i in nodes_degrees_sorted[:s]]
        x_sentinels = np.array([total_budget / s for _ in range(s)])
        test_strategy_prior.append(map_low_dim_x_to_high_dim(x_sentinels, n_nodes, indices_highest_degree_nodes))

        # ------
        # create strategy for s highest capacity nodes, budget is allocated uniformly
        indices_highest_capacity_nodes = [i[0] for i in nodes_capacities_sorted[:s]]
        x_sentinels = np.array([total_budget / s for _ in range(s)])
        test_strategy_prior.append(map_low_dim_x_to_high_dim(x_sentinels, n_nodes, indices_highest_capacity_nodes))

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

    return test_strategy_prior


def baseline(x, eval_function, node_indices, n_nodes, parallel, num_cpu_cores):
    # TODO generate initial values here instead of outside the function
    # TODO include baseline with no testing
    # TODO make this more useful somehow ??

    # TODO use the prior to create baseline values

    simulations = [100, 1000, 10000]

    x_true = np.zeros(n_nodes)
    for i, xi in zip(node_indices, x):
        x_true[i] = xi

    y = {}
    for n_sim in simulations:

        m, stderr = eval_function(x_true,
                                  n_simulations=n_sim,
                                  parallel=parallel,
                                  num_cpu_cores=num_cpu_cores)

        y[f'{n_sim}'] = [m, stderr]
    return y


def test_function(x, **kwargs):
    """
    Quadratic function just for test purposes
    @param x:
    @return:
    """
    return x[0]**2
