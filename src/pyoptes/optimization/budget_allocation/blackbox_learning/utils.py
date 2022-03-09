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


def baseline(x, eval_function, node_indices, n_nodes, statistic):
    # TODO generate initial values here instead of outside the function
    # TODO include baseline with no testing
    # TODO make this more useful somehow ??

    simulations = [100, 1000, 10000]

    x_true = np.zeros(n_nodes)
    for i, xi in zip(node_indices, x):
        x_true[i] = xi

    y = {}
    for n_sim in simulations:
        y[f'{n_sim}'] = eval_function(x_true, n_simulations=n_sim, statistic=statistic)

    return y


def test_function(x, **kwargs):
    """
    Quadratic function just for test purposes
    @param x:
    @return:
    """
    return x[0]**2
