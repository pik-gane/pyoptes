from torch.nn import Sequential
import torch
from torch import nn
from torch_scatter import scatter_mean
# from pyoptes.optimization.budget_allocation.supervised_learning.utils import Loader as Loader
# from pyoptes.optimization.budget_allocation.supervised_learning.utils import model_selection as model_selection
from typing import Optional, Tuple
from torch import Tensor
import numpy as np


class MetaLayer(torch.nn.Module):
    r"""A meta layer for building any kind of graph network, inspired by the
    `"Relational Inductive Biases, Deep Learning, and Graph Networks"
    <https://arxiv.org/abs/1806.01261>`_ paper.

    A graph network takes a graph as input and returns an updated graph as
    output (with same connectivity).
    The input graph has node features :obj:`x`, edge features :obj:`edge_attr`
    as well as global-level features :obj:`u`.
    The output graph has the same structure, but updated features.

    Edge features, node features as well as global features are updated by
    calling the modules :obj:`edge_model`, :obj:`node_model` and
    :obj:`global_model`, respectively.

    To allow for batch-wise graph processing, all callable functions take an
    additional argument :obj:`batch`, which determines the assignment of
    edges or nodes to their specific graphs.

    Args:
        edge_model (Module, optional): A callable which updates a graph's edge
            features based on its source and target node features, its current
            edge features and its global features. (default: :obj:`None`)
        node_model (Module, optional): A callable which updates a graph's node
            features based on its current node features, its graph
            connectivity, its edge features and its global features.
            (default: :obj:`None`)
        global_model (Module, optional): A callable which updates a graph's
            global features based on its node features, its graph connectivity,
            its edge features and its current global features.

    .. code-block:: python

        from torch.nn import Sequential as Seq, Linear as Lin, ReLU
        from torch_scatter import scatter_mean
        from torch_geometric.nn import MetaLayer

        class EdgeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.edge_mlp = Seq(Lin(..., ...), ReLU(), Lin(..., ...))

            def forward(self, src, dest, edge_attr, u, batch):
                # src, dest: [E, F_x], where E is the number of edges.
                # edge_attr: [E, F_e]
                # u: [B, F_u], where B is the number of graphs.
                # batch: [E] with max entry B - 1.
                out = torch.cat([src, dest, edge_attr, u[batch]], 1)
                return self.edge_mlp(out)

        class NodeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.node_mlp_1 = Seq(Lin(..., ...), ReLU(), Lin(..., ...))
                self.node_mlp_2 = Seq(Lin(..., ...), ReLU(), Lin(..., ...))

            def forward(self, x, edge_index, edge_attr, u, batch):
                # x: [N, F_x], where N is the number of nodes.
                # edge_index: [2, E] with max entry N - 1.
                # edge_attr: [E, F_e]
                # u: [B, F_u]
                # batch: [N] with max entry B - 1.
                row, col = edge_index
                out = torch.cat([x[row], edge_attr], dim=1)
                out = self.node_mlp_1(out)
                out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
                out = torch.cat([x, out, u[batch]], dim=1)
                return self.node_mlp_2(out)

        class GlobalModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.global_mlp = Seq(Lin(..., ...), ReLU(), Lin(..., ...))

            def forward(self, x, edge_index, edge_attr, u, batch):
                # x: [N, F_x], where N is the number of nodes.
                # edge_index: [2, E] with max entry N - 1.
                # edge_attr: [E, F_e]
                # u: [B, F_u]
                # batch: [N] with max entry B - 1.
                out = torch.cat([u, scatter_mean(x, batch, dim=0)], dim=1)
                return self.global_mlp(out)

        op = MetaLayer(EdgeModel(), NodeModel(), GlobalModel())
        x, edge_attr, u = op(x, edge_index, edge_attr, u, batch)
    """
    def __init__(self, edge_model=None, node_model=None, global_model=None):
        super().__init__()
        self.edge_model = edge_model
        self.node_model = node_model
        self.global_model = global_model

        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.node_model, self.edge_model, self.global_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(
            self, x: Tensor, edge_index: Tensor,
            edge_attr: Optional[Tensor] = None, u: Optional[Tensor] = None,
            batch: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """"""

        batch_size = batch

        index_sender_node = edge_index[0]
        index_receiver_node = edge_index[1]

        # print('index_sender_node', index_sender_node)
        # print('shape index_sender_node', index_sender_node.shape)
        # print('index_receiver_node', index_receiver_node)
        # print('shape index_receiver_node', index_receiver_node.shape, '\n')

        edge_batch = np.zeros((edge_attr.shape[0]))
        node_batch = np.zeros((x.shape[0]))
        
        node_batch = torch.Tensor(node_batch)
        node_batch = node_batch.long()
        
        edge_batch = torch.Tensor(edge_batch)
        edge_batch = edge_batch.long()

        # print('np shape edge batch', edge_batch.shape)
        # print('shape edge_attr', edge_attr.shape)
        # print('shape u', u.shape, '\n')

        #print(index_sender_node) #all id sender nodes
        #print(x[index_sender_node])

        if self.edge_model is not None:
            updated_edge_features = self.edge_model(x[index_sender_node], # index_src_node
                                                    x[index_receiver_node],
                                                    edge_attr,
                                                    u,
                                                    edge_batch[index_sender_node])

        if self.node_model is not None:
            updated_node_features = self.node_model(x, edge_index, updated_edge_features, u, node_batch)

        if self.global_model is not None:
            updated_graph_features = self.global_model(updated_node_features, edge_index, updated_edge_features, u, node_batch)

        return updated_node_features, updated_edge_features, updated_graph_features 

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(\n'
                f'  edge_model={self.edge_model},\n'
                f'  node_model={self.node_model},\n'
                f'  global_model={self.global_model}\n'
                f')')


class Edge_Model(torch.nn.Module):

    def __init__(self, input_features, hidden_features, output_features):
        super(Edge_Model, self).__init__()

        self.edge_mlp = Sequential(
            nn.Linear(input_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, output_features))

    def forward(self, index_src_node, index_dst_node, edge_attr, u, batch):  #source attribute, destination attribute,edge features, 
        
        #u = u.view(-1, 1) #reshape graph features because 1-dim
        
        #edge_attr = edge_attr.view(-1, 1) #reshape edge weights because 1-dim

        """G = (u, V, E)
        u = global attribute - a.e. ()
        V = {v_i} set of nodes, each v_i is a node's attribute - a.e. ()
        E = {e_k, r_k, s_k) set of edges, each e_k is the edge's attribute, r_k is 
        index of receiver node, s_k is index of sender node"""

        """torch.cat is function p - takes a set as input; 
        reduce it to single element which represents the aggregated information.
        CRUCIALLY; p functions must be invariant to permutations of their inputs - should take a 
        variable number of arguments (elementwise summation; mean; max"""

        """here the update_function is a MLP - mapped across all edges to compute per edge 
        updates; takes arguments (edge_k, node_sender, node_receiver, global_attribute.
        here the transmissions frequency between to nodes (farms)."""

        # index_src_node, index_dst_node: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.

        # print(u[301].shape)
        print('u shape', u.shape)
        print('index_src_node shape', index_src_node.shape)
        print('index_dst_node shape', index_dst_node.shape)
        print('edge_attr shape', edge_attr.shape)

        out = torch.cat([index_src_node, index_dst_node, edge_attr, u[batch]], 1) #stack source node, destination node, edge weights, graph features
        print('out shape', out.shape)
        out = self.edge_mlp(out)

        """update_function returns updated edge attributes e'k"""

        return out   


class Node_Model(torch.nn.Module):

    def __init__(self, input_features_mlp1, hidden_features_mlp1, output_features_mlp1, input_features_mlp2, hidden_features_mlp2, output_features_mlp2):
        super(Node_Model, self).__init__()
        #print(in_channels)
        self.node_mlp_1 = Sequential(
            nn.Linear(input_features_mlp1, hidden_features_mlp1),
            nn.ReLU(),
            nn.Linear(hidden_features_mlp1, hidden_features_mlp1),
            nn.ReLU(),
            nn.Linear(hidden_features_mlp1, output_features_mlp1))

        self.node_mlp_2 = Sequential(
            nn.Linear(input_features_mlp2, hidden_features_mlp2),
            nn.ReLU(),
            nn.Linear(hidden_features_mlp2, output_features_mlp2))


    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u] 
        # # batch: [N] with max entry B - 1.

        index_sender_node, index_receiver_node = edge_index

        agg_sender_nodes_edges = torch.cat([x[index_sender_node], edge_attr], dim=1) #aggregate index_sender_node with edge_attributes
    
        updated_sender_nodes = self.node_mlp_1(agg_sender_nodes_edges) #updated sender_nodes
        
        """scatter_mean: Averages all values from the src tensor into out at the indices specified in 
        the index tensor along a given axis dim. If multiple indices reference the same location, 
        their contributions average (cf. scatter_add())."""
        
        out = scatter_mean(updated_sender_nodes, index_receiver_node, dim=0, dim_size=x.size(0)) #new target feature 

        u = u.contiguous().view(-1,1) #reshape because dim=1

        agg_sender_receiver = torch.cat([x, out, u[batch]], dim=1) 
        
        updated_sender_receiver_nodes = self.node_mlp_2(agg_sender_receiver) #updated feature of target node
        #if self.residuals:
        #    out = out + edge_attr
        return updated_sender_receiver_nodes  


class Global_Model(torch.nn.Module):

    def __init__(self, input_features, hidden_features, output_features):
        super(Global_Model, self).__init__()

        self.global_mlp = Sequential(
            nn.Linear(input_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, output_features))

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.

        #u = u.view(-1, 1)
        #print(u[batch].shape)

        #print(x.shape) #120,5
        #print(batch.shape) #120
        x = scatter_mean(x, batch, dim=0) 
        #print(x.shape) #1,5

        #print(u.shape) #1,6
        u_batch = torch.zeros((u.shape[0])) #adapt to batches
        u_batch = u_batch.long() #dtype int64 for scatter_mean

        u = scatter_mean(u, u_batch, dim=0)

        out = torch.cat([u, x], dim=1) #(1,11)   5 node features, 6 graph features

        out = self.global_mlp(out) #parse into multi-layer-perceptron

        return out