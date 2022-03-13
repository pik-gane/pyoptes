from ast import Global
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.data
import warnings
from pyoptes.optimization.budget_allocation import target_function as f
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from pyoptes import set_seed
import torch
import pandas as pd
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.data import HeteroData
from pyoptes.optimization.budget_allocation.supervised_learning.utils import Loader as Loader
from pyoptes.optimization.budget_allocation.supervised_learning.utils import processing as process
from pyoptes.optimization.budget_allocation.supervised_learning.utils import model_selection as model_selection
from pyoptes.optimization.budget_allocation.supervised_learning.utils import device as get_device
from pyoptes.optimization.budget_allocation.supervised_learning.utils import training_process as train_nn
from sklearn.model_selection import train_test_split
import torch_geometric.transforms as T
from torch_geometric.utils import from_networkx
import torch_geometric
import torch_geometric.nn as geom_nn
import torch_geometric.data as geom_data
from torch_geometric.nn import GATv2Conv, LEConv, GCN2Conv, FAConv, MetaLayer, GraphConv, GINEConv, ARMAConv, SGConv, EdgeConv, GCN, GAT, GATConv, ChebConv, DenseGCNConv, GCNConv, global_mean_pool,global_add_pool, global_max_pool, SAGEConv, global_sort_pool, MLP
from torch_geometric.nn import LEConv, GCN2Conv, FAConv, MetaLayer, GraphConv, GINEConv, ARMAConv, SGConv, EdgeConv, GCN, GAT, GATConv, ChebConv, DenseGCNConv, GCNConv, global_mean_pool,global_add_pool, global_max_pool, SAGEConv, global_sort_pool, MLP
from torch_geometric.nn import LEConv, GCN2Conv, FAConv, MetaLayer, GraphConv, GINEConv, ARMAConv, SGConv, EdgeConv, GCN, GAT, GATConv, ChebConv, DenseGCNConv, GCNConv, global_mean_pool,global_add_pool, global_max_pool, SAGEConv, global_sort_pool, MLP
from torch_geometric.nn import GATv2Conv, LEConv, GCN2Conv, FAConv, MetaLayer, GraphConv, GINEConv, ARMAConv, SGConv, EdgeConv, GCN, GAT, GATConv, ChebConv, DenseGCNConv, GCNConv, global_mean_pool,global_add_pool, global_max_pool, SAGEConv, global_sort_pool, MLP
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_scatter import scatter_mean
from torch_geometric.nn import MetaLayer

def get_features(transmissions, capacities, G, time_covered):
    #print(time_covered)# 180
    edge_features = transmissions.groupby([2, 3]).size()
    edge_features.to_csv("/Users/admin/pyoptes_graphs/transmission_array.csv")
    new_df = pd.read_csv("/Users/admin/pyoptes_graphs/transmission_array.csv")

    for i in range(len(new_df)):
        _from = new_df.iloc[i][0]
        _to = new_df.iloc[i][1]
        weight = new_df.iloc[i][2]
        G[_from][_to]['weight'] = weight/time_covered  
    #adj_matrix = nx.to_numpy_matrix(G)
    #adj_list = nx.adj_matrix(G)
    #adj_list = G.adjacency_list()

    u = np.stack((f.model.p_infection_from_outside, f.model.p_infection_by_transmission, f.model.p_test_positive, 
    f.model.delta_t_testable, f.model.delta_t_infectious, f.model.delta_t_symptoms), axis = -1)

    return u, G

class Net(torch.nn.Module):
    def __init__(self, hidden_channels, dataset):
        super(Net, self).__init__()
        #torch.manual_seed(12345)
    
        #self.nn = MLP()

        #elf.conv1 = GATv2Conv(2, 16, edge_dim = 1) #simple message passing layer
        #self.conv2 = GATv2Conv(16, 32, edge_dim = 1) #simple message passing layer
        #self.conv3 = GATv2Conv(32, 64, edge_dim = 1) #simple message passing layer
        #self.conv4 = GATv2Conv(64, 128, edge_dim = 1) #simple message passing layer
        #self.conv1 = LEConv(2, 8) #simple message passing layer
        #self.conv2 = LEConv(8, 16) #simple message passing layer
        #self.conv3 = LEConv(16, 32) #simple message passing layer
        #self.conv4 = LEConv(32, 64) #simple message passing layer

        #The edge convolutional layer processes graphs or point clouds

        #SGConv, EdgeConv, ARMAConv, GINEConv, GraphConv, FAConv, GCN2Conv
        #self.conv2 = GCNConv(16, 32)
        
        #self.mlp = MLP(in_channels=128, hidden_channels=32, out_channels=32, num_layers=3)
        #self.gat = GAT(in_channels=2, hidden_channels=32, out_channels=128, num_layers=3)

        self.conv1 = GCNConv(2, 16)
        self.conv2 = GCNConv(16, 32)
        self.conv3 = GCNConv(32, 64)
    
        self.relu = nn.ReLU()
        self.elu = nn.ELU()


        """The chebyshev spectral graph convolutional operator from the
        `"Convolutional Neural Networks on Graphs with Fast Localized Spectral
        Filtering" <https://arxiv.org/abs/1606.09375>`_ paper"""
        #self.cheb1 = ChebConv(dataset.num_features, 18, K=2)
        #self.cheb2 = ChebConv(16, 8, K=2)

        self.linear = nn.Linear(64,1)
        #self.linear1 = nn.Linear(128,1)
        #self.mlp = MLP([8, 16, 8])

    def forward(self, data):
        graph_features = data.edge_attr
        x, edge_index, edge_weight, u, batch = data.x, data.edge_index, data.weight, graph_features, data.batch
        
        #Data(edge_index=[2, 430], weight=[430], num_nodes=120, x=[120, 2], y=[1], num_features=2, edge_attr=[6], num_edges=430)
        #print(edge_index[0].shape)
        #edge_index = edge_index.type(torch.LongTensor)
        #meta_layer = MetaLayer(EdgeModel())
        #edge_attr = meta_layer(x, edge_index, edge_weight, u, batch)
        #x, (edge_index, edge_weight) = self.conv1(x, edge_index, edge_weight, return_attention_weights=True)
        #x, (edge_index, edge_weight) = self.conv2(x, edge_index, edge_weight, return_attention_weights=True)
        #x, (edge_index, edge_weight) = self.conv3(x, edge_index, edge_weight, return_attention_weights=True)
        #x, (edge_index, edge_weight) = self.conv4(x, edge_index, edge_weight, return_attention_weights=True)
        #x, edges  = self.conv2(x, edge_index, edge_weight, return_attention_weights=True)
        #x, edges = self.conv3(x, edge_index, edge_weight, return_attention_weights=True)
        #x, e  = self.conv4(x, edge_index, edge_weight, return_attention_weights=True)

        h_1  = self.conv1(x, edge_index = edge_index, edge_weight = edge_weight)
        ah_1 = self.relu(h_1)
        
        h_2  = self.conv2(ah_1, edge_index = edge_index, edge_weight = edge_weight)
        ah_2 = self.relu(h_2)

        h_3  = self.conv3(ah_2, edge_index = edge_index, edge_weight = edge_weight)
        ah_3 = self.relu(h_3)

        output = self.linear(ah_3)
        #output = h_3
        #x = self.conv2(x, edge_index = edge_index, edge_weight = edge_weight)
        #x = self.relu(x)
        #x = self.conv3(x, edge_index = edge_index, edge_weight = edge_weight)
        #x = self.relu(x)
        #x = self.conv4(x, edge_index = edge_index, edge_weight = edge_weight)
        #x = self.relu(x)

        #x, edge_weight, u = self.meta_layer_3(x, edge_index, edge_weight, u, batch)
        #x = self.meta_layer_4(x, edge_index, edge_weight, u, batch)
        #print(node_features[0], edge_weight[0], edge_index[0])
        #edge_weight = self.edge1(x, edge_index)

        #x = F.dropout(x, p=0.5, training=self.training)
        #print(x.shape)
        #x = F.dropout(x, p=0.5, training=self.training)
        #x = self.relu(self.conv3(x, edge_index, edge_weight))
        #x = x.elu()
        #x = F.dropout(x, p=0.5, training=self.training)
        #x = self.conv2(x, edge_index = edge_index, edge_weight = edge_weight)

        #x = self.relu(x)
        #x = self.conv3(x, edge_index, edge_weight)
        #x = self.relu(x)
        #x = F.dropout(x, p=0.5, training=self.training)
        #x = self.conv3(x, edge_index, edge_weight)
        #    #x = self.conv5(x, edge_index, edge_weight)
        #print(x.shape)
        #x = global_mean_pool(x, data.batch) #32,64
        #x = global_max_pool(x, data.batch) #32,64
        #x = global_add_pool(x, data.batch) #32,64

        #x = global_add_pool(x, data.batch) #32,64

        #x = self.linear1(x)
        
        #x = global_add_pool(x, data.batch) #32,64
        #x = global_sort_pool(x, data.batch) #32,64
        return output #x.squeeze()