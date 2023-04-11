'''
Read and infer network (synthetic, barabasi, waxman) attributes (degree, capacity, budget share)
Attributes are saved in a csv file to be read by Graphia or Gephi
'''

import numpy as np
import os
import pandas as pd
import networkx as nx
import csv
import json
import pylab as plt


def create_graph(n: int,
                 n_nodes: int,
                 base_path: str = '../data/',
                 experiment_directory: str = ''):

    print('\n----------------------------------------')
    # get the network and its attributes
    network_path = os.path.join(base_path, f'Synset{n_nodes}-180')

    transmissions_path = os.path.join(network_path, f'syndata{n}', 'dataset.txt')
    transmissions = pd.read_csv(transmissions_path, header=None)
    transmissions = transmissions[[2, 2, 0, 1, 3]]
    edge_list = transmissions[[0, 1]]
    np_edge_list = edge_list.to_numpy().T  # transmissions sind nicht dasselbe wie die menge an edges, edges jkönnnen doppelt vorkommen

    transmissions_syn = transmissions.to_numpy()
    G = nx.from_pandas_edgelist(edge_list, source=0, target=1, create_using=nx.DiGraph())
    degrees_syn = np.array(list(G.degree()))

    edge_index = G.edges

    # # incoming edges per node
    # f = G.in_edges()
    # in_edge_per_node = []
    # for node in range(n_nodes):
    #     in_edge_per_node.append([node, 0])
    #     for i, o in f:
    #         if i == node:
    #             in_edge_per_node[node][1] += 1
    #
    # # outgoing edges per node
    # ff = G.out_edges()
    # out_edge_per_node = []
    # for node in range(n_nodes):
    #     out_edge_per_node.append([node, 0])
    #     for i, o in ff:
    #         if i == node:
    #             out_edge_per_node[node][1] += 1
    #
    # # incoming and outgoing edges per node
    # total_edge_per_node = []
    # for i, _ in enumerate(in_edge_per_node):
    #     total_edge_per_node.append([in_edge_per_node[i][0], in_edge_per_node[i][1] + out_edge_per_node[i][1]])

    # some nodes are slaughterhouses which have no trading partners, only incoming transmissions
    # Consequently, they don't exist in the network although they are needed for the simulation
    # they get assigned a degree of 0
    if np.shape(degrees_syn)[0] < n_nodes:
        # get the indices of the missing nodes
        all_node_indices = list(range(n_nodes))
        missing_node_indices = list(set(all_node_indices) - set(degrees_syn[:, 0]))
        # assign the missing nodes a degree of 0
        missing_degrees = np.array([[i, 0] for i in missing_node_indices])
        # join the missing degrees with the existing degrees
        degrees_syn = np.concatenate((degrees_syn, missing_degrees), axis=0)

    capacities_path = os.path.join(network_path, f'syndata{n}', 'barn_size.txt')
    capacities = pd.read_csv(capacities_path, header=None)
    capacities_syn = capacities.iloc[0][:n_nodes].to_numpy()
    # node_capacities contains only capacities, add node_index sort nodes by capacities and get their indices
    # capacities_syn = [(i, c) for i, c in enumerate(capacities_syn)]
    transmissions = transmissions_syn
    capacities = capacities_syn
    degrees = degrees_syn

    # get the budget for the optimizer
    path_best_strategy = os.path.join(experiment_directory, f'individual/{n}', 'best_parameter.npy')
    best_strategy = np.load(path_best_strategy)
    # replace the values in best_strategy smaller or equal than the median by 0
    best_strategy[best_strategy <= 1e-4] = 0
    # print('max,min, mean, median, sum budget share', np.max(best_strategy), np.min(best_strategy),
    #       np.mean(best_strategy), np.median(best_strategy), np.sum(best_strategy))
    # print(np.histogram(best_strategy, bins=10))
    print(np.shape(transmissions), 'transmissions')
    print(np.shape(capacities), 'capacities')
    print(np.shape(degrees), 'degrees')

    # get experiment specific hyperparameters
    with open(os.path.join(experiment_directory, 'experiment_hyperparameters.json'), 'r') as f:
        hyperparameters = json.load(f)

    optimizer = hyperparameters['optimizer_hyperparameters']['optimizer']

    print(np.shape(edge_index), 'edge_index')
    # csv containing the edges of the network
    with open(f'../data/blackbox_learning/budget/edges_{optimizer}_{n_nodes}_nodes_network_{n}.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows([['Source', 'Target']])
        writer.writerows(edge_index)

    print('shape degrees', np.shape(degrees))
    print('shape capacities', np.shape(capacities))
    print('shape best strategy', np.shape(best_strategy))
    print(np.histogram(best_strategy))

    # sort degrees ascending
    degrees = degrees[degrees[:, 0].argsort()]

    with open(f'../data/blackbox_learning/budget/attributes_{optimizer}_{n_nodes}_nodes_network_{n}.csv', 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows([('node_id', 'Budget_share', 'degree', 'capacity')])
        for i in range(n_nodes):
            writer.writerows([(degrees[i][0], best_strategy[i], degrees[i][1], capacities[i])])


if __name__ == '__main__':
    # for all three optimizers
    # find examples for budgets better than the baseline
    # find examples for budgets worse than the baseline
    #

    network_path = '../../networks/data/'
    base_path = '../data/blackbox_learning/results'

    # experiment = '20230109_cma_mean_nodes_120'
    # experiment_directory = os.path.join(base_path, experiment)
    #
    # create_graph(n=0,
    #              n_nodes=120,
    #              base_path=network_path,
    #              experiment_directory=experiment_directory)
    #
    # # ---------------------
    # experiment = '20230120_np_mean_nodes_120'
    # experiment_directory = os.path.join(base_path, experiment)
    #
    # create_graph(n=0,
    #              n_nodes=120,
    #              base_path=network_path,
    #              experiment_directory=experiment_directory)
    #
    # # ---------------------
    # experiment = '20230226_gpgo_mean_nodes_120'
    # experiment_directory = os.path.join(base_path, experiment)
    #
    # create_graph(n=0,
    #              n_nodes=120,
    #              base_path=network_path,
    #              experiment_directory=experiment_directory)

    experiment = '20230109_cma_mean_nodes_1040'
    experiment_directory = os.path.join(base_path, experiment)

    # create_graph(n=1,
    #                 n_nodes=1040,
    #                 base_path=network_path,
    #                 experiment_directory=experiment_directory)

    experiment = '20230226_gpgo_rms_nodes_57590_sentinels_1329'
    experiment_directory = os.path.join(base_path, experiment)

    create_graph(n=2,
                    n_nodes=57590,
                    base_path=network_path,
                    experiment_directory=experiment_directory)


