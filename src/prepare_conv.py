import networkx as nx
import warnings
import numpy as np
from pyoptes import set_seed
from pyoptes.optimization.budget_allocation import target_function as f
from gnn import get_features
import pandas as pd
from torch_geometric.utils import from_networkx
import torch


def prepare_convolutions(x,y):
    #print(test_x.shape)
    warnings.filterwarnings("ignore")

    #SI model params
    n_trials = 1
    evaluation_parms = { 
            'n_simulations': 100, 
            'statistic': lambda a: np.mean(a**2)
            }

    # set some seed to get reproducible results:
    set_seed(1)

    # generate a Waxman graph:
    waxman = nx.waxman_graph(120)
    pos = dict(waxman.nodes.data('pos'))
    # convert into a directed graph:
    G = nx.DiGraph(nx.to_numpy_array(waxman))

    #prepare SI model; target function
    f.prepare(
    use_real_data=False, 
    static_network=G,
    n_nodes=120,
    max_t=365, 
    expected_time_of_first_infection=30,#
    capacity_distribution = np.random.lognormal, #lambda size: np.ones(size), # any function accepting a 'size=' parameter
    delta_t_symptoms=60 #time until show symptoms 
    )

    #store SI model parameters
    n_inputs = f.get_n_inputs()
    capacities = f.capacities
    time_covered = f.transmissions_time_covered
    total_budget = n_inputs

    transmissions_csv = pd.DataFrame(f.model.transmissions_array, index = None)
    #get features of graph

    graph_features, G = get_features(transmissions_csv, capacities, G, time_covered)

    #create dataset
    #print(edge_features_m)
    #edge_f = nx.convert_matrix.from_pandas_edgelist(edge_features_m)
    #print(edge_f)
    #edge_index = [e for e in G.edges]
    #edge_index = torch.LongTensor(edge_index
    #data = Data(x =node_features, edge_index=edge_index, edge_attr = G.["weights"])
    #data = HeteroData()

    data_list = []
    for i in range(len(x[0])):
        dataset = from_networkx(G)
        dataset.x = torch.from_numpy(np.stack((x.iloc[i].to_numpy(), capacities), axis = -1))
        #dataset.x = torch.from_numpy(x.iloc[i].to_numpy()).reshape(120,1)
        dataset.y =  torch.from_numpy(y.iloc[i].to_numpy()) 
        dataset.num_nodes = dataset.x.shape[0]
        dataset.num_features = len(dataset.x.shape)
        dataset.edge_attr = torch.from_numpy(graph_features)
        dataset.num_edges = len(dataset.weight)
        data_list.append(dataset)

    #gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
    #              normalization_out='col',
    #              diffusion_kwargs=dict(method='ppr', alpha=0.05),
    #              sparsification_kwargs=dict(method='topk', k=128,
    #                                         dim=0), exact=True)
    #dataset = gdc(dataset)
    #np.mean(np.array([f.evaluate(x, **evaluation_parms) for it in range(n_trials)]))
    print(data_list[0])
    return data_list