import networkx as nx
import warnings
import numpy as np
from pyoptes import set_seed
from pyoptes.optimization.budget_allocation import target_function as f
from gnn import get_features
import pandas as pd
from torch_geometric.utils import from_networkx
import torch
from pyoptes import set_seed
from scipy.stats import lognorm


def caps(size): 
  return lognorm.rvs(s=2, scale=np.exp(4), size=size)


def prepare_convolutions(x,y):
    set_seed(1)
    #print(test_x.shape)
    warnings.filterwarnings("ignore")
    n_nodes = 120
    # set some seed to get reproducible results:
    # generate a Waxman graph:
    waxman = nx.waxman_graph(n_nodes)
    pos = dict(waxman.nodes.data('pos'))
    # convert into a directed graph:
    G = nx.DiGraph(nx.to_numpy_array(waxman))

    #SI model params
    f.prepare(
    use_real_data=False, #False = synthetic data
    static_network=G, #use waxman graph
    n_nodes=n_nodes, #size of network
    max_t=365, #time horizon
    expected_time_of_first_infection=30, #30 days
    capacity_distribution = caps, #lambda size: np.ones(size), # any function accepting a 'size=' parameter
    delta_t_symptoms=60 #abort simulation after 60 days when symptoms first show up and testing becomes obsolet
    )

    #SI models param
    n_inputs = f.get_n_inputs() #number of nodes of our network
    capacities = f.capacities
    time_covered = f.transmissions_time_covered
    total_budget = n_inputs
    print("n_inputs (=number of network nodes):", n_inputs)

    total_budget = 1.0 * n_inputs #total budget equals number of nodes
    n_simulations = 10000 #run n_simulations of our target function to reduce std error
    num_cpu_cores = -1 #use all cpu cores

    #evaluation params of our target function
    evaluation_parms = { 
            'n_simulations': n_simulations, 
            'parallel': True,
            'num_cpu_cores': num_cpu_cores
            }
    
  
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