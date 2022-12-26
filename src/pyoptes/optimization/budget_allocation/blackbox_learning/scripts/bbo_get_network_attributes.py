'''
Read and infer network (synthetic, barabasi, waxman) attributes (degree, capacity, budget share)
Attributes are saved in a csv file to be read by Graphia or Gephi
'''

import numpy as np
import os
import pandas as pd
import networkx as nx
import csv


def create_graph(n, n_nodes, network_base_path, budget_path='plots'):
    """
    Loads n_runs graphs from disk and returns them as a list
    @param budget_path:
    @param n_nodes:
    @param network_base_path:
    @param n: int, the number of the graph to be loaded
    @return: three lists, containing transmissions, capacities and degrees for the loaded graph respectively
            all lists are sorted by node index
    """

    network_path = os.path.join(base_path, f'Synset{n_nodes}-180')
    # print(f'Loading synthetic graph number {n}')

    transmissions_path = os.path.join(network_path, f'syndata{n}', 'dataset.txt')
    transmissions = pd.read_csv(transmissions_path, header=None)
    transmissions = transmissions[[2, 2, 0, 1, 3]]
    edge_list = transmissions[[0, 1]]
    transmissions_syn = transmissions.to_numpy()

    capacities_path = os.path.join(network_path, f'syndata{n}', 'barn_size.txt')
    capacities = pd.read_csv(capacities_path, header=None)

    print('capacities', np.shape(capacities))

    G = nx.from_pandas_edgelist(edge_list, source=0, target=1, create_using=nx.DiGraph())

    # # Lists the edges of the graph as a list of tuples. Tuples contain two indices each
    # with open(f'../../network{n}_{n_nodes}_edges.csv', 'w', newline='') as f:
    #     writer = csv.writer(f, delimiter=',')
    #     writer.writerows(G.edges)
    print('degree', np.shape(G.degree))



    print(dfgs)

    # TODO budget share has to be taken from another file, probably should be a parameter
    with open(f'../../network{n}_{n_nodes}_attributes.csv', 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows([('Budget_share', 'degree', 'capacity'), (1,2,3)])
        # writer.writerows(G.edges)
    print(G)


base_path = '../../../../../../../networks/data/'

create_graph(0, 1040, network_base_path=base_path)

create_graph(0, 57590, network_base_path=base_path)

