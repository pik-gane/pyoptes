from pyoptes.optimization.budget_allocation.blackbox_learning.GNN_neural_process.Meta_Layer import MetaLayer
from pyoptes.optimization.budget_allocation.blackbox_learning.GNN_neural_process.Meta_Layer import Node_Model, Edge_Model, Global_Model
from pyoptes.optimization.budget_allocation import target_function as f

import numpy as np
import argparse
from tqdm import tqdm
import torch
import networkx as nx
import os
import pandas as pd

def create_graph(n: int,
                 graph_type: str,
                 n_nodes: int,
                 base_path: str = '../data/',
                 return_edge_list: bool = False) -> tuple:
    """
    Loads n_runs graphs from disk and returns them as a list
    @param return_edge_list:
    @param n_nodes:
    @param base_path:
    @param n: int, the number of the graph to be loaded
    @param graph_type: string, barabasi-albert, waxman or synthetic
    @return: three lists, containing transmissions, capacities and degrees for the loaded graph respectively
            all lists are sorted by node index
    """
    if graph_type == 'waxman':
        network_path = os.path.join(base_path, graph_type + '_networks', f'{n_nodes}')
        # print(f'Loading waxman graph number {n}')

        transmission_path = os.path.join(network_path, f'WX{n}', 'transmissions.txt')
        transmissions_waxman = pd.read_csv(transmission_path, header=None).to_numpy()

        capacities_path = os.path.join(network_path, f'WX{n}', 'capacity.txt')
        capacities_waxman = np.int_(pd.read_csv(capacities_path, header=None).to_numpy().squeeze())
        # node_capacities contains only capacities, add node_index sort nodes by capacities and get their indices
        # capacities_waxman = [(i, c) for i, c in enumerate(capacities_waxman)]

        degrees_path = os.path.join(network_path, f'WX{n}', 'degree.txt')
        degrees_waxman = pd.read_csv(degrees_path, header=None).to_numpy()

        transmissions = transmissions_waxman
        capacities = capacities_waxman
        degrees = degrees_waxman

    elif graph_type == 'ba':
        network_path = os.path.join(base_path, graph_type + '_networks', f'{n_nodes}')
        # print(f'Loading barabasi-albert graph number {n}')

        single_transmission_path = os.path.join(network_path, f'BA{n}', 'transmissions.txt')
        transmissions_ba = pd.read_csv(single_transmission_path, header=None).to_numpy()

        capacities_path = os.path.join(network_path, f'BA{n}', 'capacity.txt')
        capacities_ba = np.int_(pd.read_csv(capacities_path, header=None).to_numpy().squeeze())
        # node_capacities contains only capacities, add node_index sort nodes by capacities and get their indices
        # capacities_ba = [(i, c) for i, c in enumerate(capacities_ba)]

        degrees_path = os.path.join(network_path, f'BA{n}', 'degree.txt')
        degrees_ba = pd.read_csv(degrees_path, header=None).to_numpy()

        transmissions = transmissions_ba
        capacities = capacities_ba
        degrees = degrees_ba

    elif graph_type == 'syn':
        network_path = os.path.join(base_path, f'Synset{n_nodes}-180')
        # print(f'Loading synthetic graph number {n}')

        transmissions_path = os.path.join(network_path, f'syndata{n}', 'dataset.txt')
        transmissions = pd.read_csv(transmissions_path, header=None)
        transmissions = transmissions[[2, 2, 0, 1, 3]]
        edge_list = transmissions[[0, 1]]
        np_edge_list = edge_list.to_numpy().T   # transmissions sind nicht dasselbe wie die menge and edges, edges jkönnnen doppelt vorkommen

        transmissions_syn = transmissions.to_numpy()

        G = nx.from_pandas_edgelist(edge_list, source=0, target=1, create_using=nx.DiGraph())
        degrees_syn = np.array(list(G.degree()))
        f = G.in_edges()
        in_edge = []
        for node in range(n_nodes):
            in_edge.append([node, 0])
            for i, o in f:
                # print(i, o)
                if i == node:
                    in_edge[node][1] += 1
        # print(in_edge)
        # print('shape of f', np.shape(f))
        ff = G.out_edges()
        out_edge = []
        for node in range(n_nodes):
            out_edge.append([node, 0])
            for i, o in ff:
                # print(i, o)
                if i == node:
                    out_edge[node][1] += 1
        # print(out_edge)
        # print('np shape out_edge', np.shape(out_edge))

        total_edge = []
        for i, _ in  enumerate(in_edge):
            total_edge.append([in_edge[i][0], in_edge[i][1] + out_edge[i][1]])

        # print('total_edge', total_edge)
        # print('np shape total_edge', np.shape(total_edge))
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

    else:
        Exception(f'Graph type {graph_type} not supported')

    if return_edge_list:
        return transmissions, capacities, degrees, np_edge_list, in_edge, out_edge, total_edge
    else:
        return transmissions, capacities, degrees


def removeDuplicates(lst):
    """
    Remove duplicate elements from a list
    @param lst:
    @return:
    """
    return list(set([i for i in lst]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--sentinels", type=int, default=1040,
                        help="Set the number of nodes that are used. Has to be smaller than or equal to n_nodes. "
                             "Default is 120 nodes.")
    parser.add_argument("--n_nodes", type=int, default=1040, choices=[120, 1040, 57590],
                        help="Si-simulation parameter. "
                             "Defines the number of nodes used by the SI-model to create a graph. "
                             "Default value is 120 nodes.")
    # ------------------ SI-simulation hyperparameters ------------------
    parser.add_argument("--n_simulations", type=int, default=10000,
                        help="Si-simulation parameter. Sets the number of runs the for the SI-model. "
                             "Higher values of n_simulations lower the variance of the output of the simulation. "
                             "Default value is 1000.")
    parser.add_argument('--graph_type', choices=['waxman', 'ba', 'syn'], default='syn',
                        help='Si-simulation parameter. Set the type of graph the simulation uses.'
                             ' Either Waxman,Synthetic or Barabasi-Albert (ba) can be used. Default is Synthetic.')

    parser.add_argument('--delta_t_symptoms', type=int, default=60,
                        help='Si-simulation parameter.. Sets the time (in days) after which an infection is detected'
                             ' automatically. Default is 60 days')
    parser.add_argument('--p_infection_by_transmission', type=float, default=0.5,
                        help='Si-simulation parameter. The probability of how likely a trade animal '
                             'infects other animals. Default is 0.5.')
    parser.add_argument('--expected_time_of_first_infection', type=int, default=30,
                        help='Si-simulation parameter. '
                             'The expected time (in days) after which the first infection occurs. ')
    parser.add_argument('--delta_t_infectious', type=int, default=0,
                        help='')
    parser.add_argument('--delta_t_testable', type=int, default=0,
                        help='')
    parser.add_argument('--p_test_positive', type=float, default=0.90,
                        help='Si-simulation parameter. The probability of how likely a test is positive. Default is 0.9.')
    # ------------------ utility hyperparameters ------------------
    parser.add_argument('--mode_choose_sentinels', choices=['degree', 'capacity', 'transmission'], default='degree',
                        help='Sets the mode of how sentinels are chosen. ')
    parser.add_argument('--path_networks', default='../../networks/data',
                        help='Location where the networks are saved to. '
                             'Path on cluster. /p/projects/ou/labs/gane/optes/mcmc_100nets/data'
                             '/p/projects/ou/labs/gane/optes/mcmc_100nets/data/')
    args = parser.parse_args()

    transmissions, capacities, degrees, edge_list, \
    in_edge_per_node, out_edge_per_node, total_edge_per_node = create_graph(n=0,
                                                                             graph_type=args.graph_type,
                                                                             n_nodes=args.n_nodes,
                                                                             base_path=args.path_networks,
                                                                             return_edge_list=True)

    print('edge_list', edge_list)
    print('np shape', np.shape(edge_list), '\n')

    total_budget = args.n_nodes

    x = np.ones(args.sentinels) * total_budget / args.sentinels

    ml = MetaLayer(
        Edge_Model(
            input_features=27, hidden_features=128, output_features=1),
            # TODO list number of input features
        Node_Model(
            input_features_mlp1=11, hidden_features_mlp1=128, output_features_mlp1=128,
            input_features_mlp2=139, hidden_features_mlp2=256, output_features_mlp2=6),
        Global_Model(
            input_features=12, hidden_features=128, output_features=5)
    )

    # inti si-model and create an edge index (list of all transmission pairs)

    # initialize the si-simulation
    f.prepare(n_nodes=args.n_nodes,
              capacity_distribution=capacities,
              pre_transmissions=transmissions,
              p_infection_by_transmission=args.p_infection_by_transmission,
              delta_t_symptoms=args.delta_t_symptoms,
              expected_time_of_first_infection=args.expected_time_of_first_infection,
              static_network=None,
              use_real_data=False)

    # create edge_index for later use in GNN
    source_nodes = f.model.transmissions_array[:, 2]
    destination_nodes = f.model.transmissions_array[:, 3]

    edge_index = list(zip(source_nodes, destination_nodes))
    edge_index = removeDuplicates(edge_index)

    budget_share = torch.tensor(x.reshape(np.shape(x)[0], 1))
    cap = torch.tensor(capacities.reshape(np.shape(capacities)[0], 1))
    deg = torch.tensor(degrees)
    in_edge_per_node = torch.tensor(in_edge_per_node)
    out_edge_per_node = torch.tensor(out_edge_per_node)
    total_edge_per_node = torch.tensor(total_edge_per_node)

    # print('shape budget_share', budget_share.shape)
    # print('shape cap', cap.shape)
    # print('shape deg', deg.shape)
    print('shape edge', in_edge_per_node.shape)
    print('shape out_edge', out_edge_per_node.shape)
    print('shape total_edge', total_edge_per_node.shape)

    node_features = torch.cat((budget_share,
                              cap,
                              deg,
                              in_edge_per_node, # incoming_transmissions
                              out_edge_per_node, # outgoing_transmissions
                              total_edge_per_node), dim=1)
    node_features = torch.tensor(node_features, dtype=torch.float) # TODO creates a warning for dtype

    print('node_features shape', node_features.shape)

    # edge_attributes [number_of_edges, number_of_features]
    # für jedes edge die menge an transmissions an dem edge -> ein feature pro edge
    edge_attributes = []

    all_edges = list(zip(source_nodes, destination_nodes))
    for edge in edge_index:

        transmissions_per_edge = len(list(filter([edge].__contains__, all_edges)))
        edge_attributes.append(transmissions_per_edge)

    edge_attributes = torch.tensor(np.array(edge_attributes).reshape(np.shape(edge_attributes)[0], 1), dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).T
    print('np shape edge_index', np.shape(edge_index), '\n')

    print(f'edge_attributes target shape: [number_of_edges, number_of_features] ()')
    print('np shape edge_attributes', np.shape(edge_attributes), '\n')
    #
    # stack graph_features derived from SI model parameters defined above
    p_infection_from_outside = 1 / (args.n_nodes * args.expected_time_of_first_infection)
    p_test_positive = args.p_test_positive
    delta_t_testable = args.delta_t_testable
    delta_t_infectious = args.delta_t_infectious
    delta_t_symptoms = args.delta_t_symptoms
    p_infection_by_transmission = args.p_infection_by_transmission

    # shape (1,6)
    graph_features = torch.tensor(np.stack(
                                (p_infection_from_outside,
                                 p_infection_by_transmission,
                                 p_test_positive,
                                 delta_t_testable,
                                 delta_t_infectious,
                                 delta_t_symptoms), axis=-1).reshape(1, 6), dtype=torch.float)

    print('graph_features', graph_features)
    print('shape graph_features', graph_features.shape, '\n')

    print('------------')

    print('created meta layer')
    ml.reset_parameters()
    print('reset parameters')
    updated_node_features, updated_edge_features, updated_graph_features = ml.forward(x=node_features,
                                                                                      edge_index=edge_index,
                                                                                      edge_attr=edge_attributes,
                                                                                      u=graph_features)

    print('updated_node_features', updated_node_features)
    print('shape updated_node_features', updated_node_features.shape, '\n')
    print('updated_edge_features', updated_edge_features)
    print('shape updated_edge_features', updated_edge_features.shape, '\n')
    print('updated_graph_features', updated_graph_features)
    print('shape updated_graph_features', updated_graph_features.shape, '\n')

    print('------------')
    print('node features')
    budget_share = updated_node_features[:, 0]
    print('budget_share', budget_share)
    print('shape budget_share', budget_share.shape)
    print('sum, min, max', torch.sum(budget_share), torch.min(budget_share), torch.max(budget_share), '\n')
    # TODO results are wildly incorrect, not sure if that is because of no training or missing softmax

    # create a training dataset and loop
    # für batch_sizes > 1, müssen die einzelnen graphen mit cat zusammengefügt werden

    features = np.concatenate((degrees,
                               capacities,
                               in_edge_per_node,
                               out_edge_per_node,
                               total_edge_per_node,
                               graph_features,
                               edge_attributes,
                               source_nodes,
                               destination_nodes), axis=None).reshape(1, -1)

    print(features.shape)

    initial_y = 500.0
    y = 1000.0
    batch = [features, y]

    print(np.shape(batch))

    # TODO some kinda loss function
    loss = initial_y - y



    # shapes can be correctly extracted if the shape of the edge attributes is passed
    # print('np.shape degrees', np.shape(degrees))
    # print('np.shape capacities', np.shape(capacities))
    #
    # print('np.shape in_edge_per_node', np.shape(in_edge_per_node))
    # print('np.shape out_edge_per_node', np.shape(out_edge_per_node))
    # print('np.shape total_edge_per_node', np.shape(total_edge_per_node))
    #
    # print('np.shape graph_features', np.shape(graph_features))
    # print('np.shape edge_attributes', np.shape(edge_attributes))
    # print('np.shape source_nodes', np.shape(source_nodes))
    # print('np.shape destination_nodes', np.shape(destination_nodes), '\n')
    #
    # print('\n', 'current_state', features)
    # print('shape current_state', features.shape, '\n')

    # degrees = np.reshape(features[:, :args.n_nodes*2], (args.n_nodes, 2))
    # print('np shape degrees', np.shape(degrees))
    # capacities = np.reshape(features[:, args.n_nodes*2:args.n_nodes*3], (args.n_nodes))
    # print('np shape capacities', np.shape(capacities))
    #
    # incoming_transmissions = np.reshape(features[:, args.n_nodes * 3:args.n_nodes * 5], (args.n_nodes, 2))
    # print('np shape incoming_transmissions', np.shape(incoming_transmissions))
    # outgoing_transmissions = np.reshape(features[:, args.n_nodes * 5:args.n_nodes * 7], (args.n_nodes, 2))
    # print('np shape outgoing_transmissions', np.shape(outgoing_transmissions))
    # total_transmissions = np.reshape(features[:, args.n_nodes * 7:args.n_nodes * 9], (args.n_nodes, 2))
    # print('np shape total_transmissions', np.shape(total_transmissions))
    # graph_features = features[:, args.n_nodes * 9:args.n_nodes * 9 + 6]
    # print('np shape graph_features', np.shape(graph_features))
    #
    # no_edgeattributes = np.shape(edge_attributes)[0]
    # print('no_edgeattributes', no_edgeattributes)
    # edge_attributes = np.reshape(features[:, args.n_nodes * 9 + 6:
    #                               args.n_nodes * 9 + 6 + no_edgeattributes], (-1, 1))
    # print('np shape edge_attributes', np.shape(edge_attributes))
    #
    # source_nodes = np.reshape(features[:, args.n_nodes * 9 + 6 + no_edgeattributes:
    #                            args.n_nodes * 9 + 6 + no_edgeattributes + np.shape(source_nodes)[0]], (-1, 1))
    #
    # print('np shape source_nodes', np.shape(source_nodes))
    # destination_nodes = np.reshape(features[:,
    #                                args.n_nodes * 9 + 6 + no_edgeattributes + np.shape(source_nodes)[0]:
    #                                args.n_nodes * 9 + 6 + no_edgeattributes + np.shape(source_nodes)[0] + np.shape(destination_nodes)[0]], (-1, 1))
    # print('np shape destination_nodes', np.shape(destination_nodes))


    # if batch gleich 1 dann shape (1,state), bei 4 -> (4, state)
    # wird an first line in custom policy übergeben
