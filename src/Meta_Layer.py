from torch.nn import Sequential
import torch
from torch import nn
from torch_scatter import scatter_mean
from torch_geometric.nn import MetaLayer
from pyoptes.optimization.budget_allocation.supervised_learning.utils import Loader as Loader
from pyoptes.optimization.budget_allocation.supervised_learning.utils import model_selection as model_selection

class Edge_Model(torch.nn.Module):

    def __init__(self, ins, hiddens, outs):
        super(Edge_Model, self).__init__()

        self.edge_mlp = Sequential(
            nn.Linear(ins+5, hiddens),
            nn.ReLU(),
            nn.Linear(hiddens, hiddens),
            nn.ReLU(),
            nn.Linear(hiddens, ins))

    def forward(self, src_node, dst_node, edge_attr, u, batch):  #source attribute, destination attribute,edge features, 
        
        u = u.view(-1, 1) #reshape graph features

        edge_attr = edge_attr.view(-1, 1) #reshape edge weights

        out = torch.cat([src_node, dst_node, edge_attr, u[batch]], 1) #stack source node, destination node, edge weights, graph features

        out = self.edge_mlp(out)

        return out   

class Node_Model(torch.nn.Module):

    def __init__(self, ins, hiddens, outs):
        super(Node_Model, self).__init__()
        #print(in_channels)
        self.node_mlp_1 = Sequential(
            nn.Linear(ins+1, hiddens),
            nn.ReLU(),
            nn.Linear(hiddens, hiddens),
            nn.ReLU(),
            nn.Linear(hiddens, hiddens))
        self.node_mlp_2 = Sequential(
            nn.Linear(hiddens+3, hiddens+3),
            nn.ReLU(),
            nn.Linear(hiddens+3, ins))


    def forward(self, x, edge_index, edge_attr, u, batch):
        src_node, dest = edge_index
        
        out = torch.cat([x[src_node], edge_attr], dim=1) #updated edge // stack all edge features 
        
        #print(out.shape, x[src_node].shape, x[dest].shape)
        out = self.node_mlp_1(out) #updated feature of our edges
        
        """scatter_mean: Averages all values from the src tensor into out at the indices specified in the index tensor 
        along a given axis dim. If multiple indices reference the same location, 
        their contributions average (cf. scatter_add())."""
        
        out = scatter_mean(out, dest, dim=0, dim_size=x.size(0)) #new target feature 

        u = u.view(-1,1)
        out = torch.cat([x, out, u[batch]], dim=1) 
        
        out = self.node_mlp_2(out) #updated feature of target node
        #if self.residuals:
        #    out = out + edge_attr
        return out  

#src_node = source, dest = target, node_model = target node model, 

class Global_Model(torch.nn.Module):

    def __init__(self, ins, hiddens, outs):
        super(Global_Model, self).__init__()

        self.global_mlp = Sequential(
            nn.Linear(ins+2, hiddens),
            nn.ReLU(),
            nn.Linear(hiddens, hiddens),
            nn.ReLU(),
            nn.Linear(hiddens, outs))

    def forward(self, x, edge_index, edge_attr, u, batch):

        u = u.view(-1, 6) #reshape graph features

        x = scatter_mean(x, batch, dim=0)

        out = torch.cat([u, x], dim=1) #stack graph features, x horizontally

        out = self.global_mlp(out) #parse into multi-layer-perceptron

        return out

class meta_layer(nn.Module):
    """Graph Netural Network"""
    def __init__(self, ins_nodes, ins_edges, ins_graphs, hiddens, outs):
        super(meta_layer, self).__init__()

        self.layer_1 = MetaLayer(Edge_Model(ins_edges, hiddens, outs),
                            Node_Model(ins_nodes, hiddens, outs),
                            Global_Model(ins_graphs, hiddens, outs))

        self.layer_2 = MetaLayer(Edge_Model(ins_edges, hiddens, outs),
                            Node_Model(ins_nodes, hiddens, outs),
                            Global_Model(ins_graphs, hiddens, outs))

        self.layer_3 = MetaLayer(Edge_Model(ins_edges, hiddens, outs),
                            Node_Model(ins_nodes, hiddens, outs),
                            Global_Model(ins_graphs, hiddens, outs))

        self.layer_4 = MetaLayer(Edge_Model(ins_edges, hiddens, outs),
                            Node_Model(ins_nodes, hiddens, outs),
                            Global_Model(ins_graphs, hiddens, outs-5))
        
        self.linear = nn.Linear(ins_nodes, 1)

        #self.layer_5 = MetaLayer(Global_Model(ins_graphs, hiddens, outs=outs-5))

    # Defining the forward pass    
    def forward(self, x, edge_attr, u, edge_index, batch):
        
        hx_1, h1_edge_attr, hu_1 = self.layer_1(x=x, edge_attr=edge_attr, edge_index=edge_index, u=u, batch=batch)

        for i in range(5):

            hx_2, h2_edge_attr, hu_2 = self.layer_2(x=hx_1, edge_attr=h1_edge_attr, edge_index=edge_index, u=hu_1, batch=batch)
            hx_1 = hx_2 
            h1_edge_attr = h2_edge_attr
            hu_2 = hu_1

        hx_3, h3_edge_attr, hu_3 = self.layer_3(x=hx_2, edge_attr=h2_edge_attr, edge_index=edge_index, u=hu_2, batch=batch)

        hx_4, h4_edge_attr, hu_4 = self.layer_4(x=hx_3, edge_attr=h3_edge_attr, edge_index=edge_index, u=hu_3, batch=batch)
        
        out_x = self.linear(hx_4)

        return out_x, h4_edge_attr, hu_4
