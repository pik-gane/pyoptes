import numpy as np
import matplotlib.pyplot


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


def baseline(x, eval_function, node_indices, n_nodes, statistic):

    simulations = [100, 1000, 10000]

    x_true = np.zeros(n_nodes)
    for i, xi in zip(node_indices, x):
        x_true[i] = xi

    y = {}
    for n_sim in simulations:
        y[f'{n_sim}'] = eval_function(x_true, n_simulations=n_sim, statistic=statistic)

    return y