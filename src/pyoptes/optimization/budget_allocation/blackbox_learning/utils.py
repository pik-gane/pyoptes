import os
import json
import numpy as np
import matplotlib.pyplot


def test_function(x, n_simulations, node_indices, n_nodes, eval_function, statistic, total_budget):
    """

    @param x:
    @return:
    """
    return x[0]**2


def save_results(model_return, model_steps, model_hits, base_path):
    """
    Saves the results for a training run of a model,
    Results are saved as numpy files.
    Args:

        model_return: list,
        model_steps: list
        model_hits: list
        base_path: string, base path where the results are saved to
    """
    if not os.path.isdir(base_path):
        os.makedirs(base_path)

    np.save(os.path.join(base_path, 'model_steps'), model_steps)


def save_parameters(parameters, base_path):
    """
    Saves the parameters for a training run of a model,
    Parameters are saved as a json file.
    Args:
        parameters: dict,
        base_path: string, base path where the results are saved to
    """
    if not os.path.isdir(base_path):
        os.makedirs(base_path)

    json_path = os.path.join(base_path, 'experiment_parameters.json')

    with open(json_path, 'w') as fp:
        json.dump(parameters, fp, sort_keys=True, indent=4)


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


def baseline(x, eval_function, node_indices, n_nodes, statistic):
    # TODO generate initial values here instead of outside the function
    # TODO include baseline with no testing

    simulations = [100, 1000, 10000]

    x_true = np.zeros(n_nodes)
    for i, xi in zip(node_indices, x):
        x_true[i] = xi

    y = {}
    for n_sim in simulations:
        y[f'{n_sim}'] = eval_function(x_true, n_simulations=n_sim, statistic=statistic)

    return y