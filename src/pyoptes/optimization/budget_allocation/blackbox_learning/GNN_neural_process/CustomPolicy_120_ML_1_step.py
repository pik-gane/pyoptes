from typing import Callable, Dict, List, Optional, Tuple, Type, Union
import gym
import torch as th
from torch import batch_norm, nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from torch.nn import Linear, ReLU
from torch_geometric.nn import Sequential, GCNConv, GATConv
import pandas as pd
import numpy as np
import torch
from torch_geometric.nn import global_mean_pool
from typing import List, Optional, Union
from Meta_Layer import MetaLayer, Edge_Model, Node_Model, Global_Model
import torch
from torch import Tensor
from torch_scatter import scatter
from torchinfo import summary
import networkx as nx
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_scatter import scatter_mean

"""
model = Sequential('x, edge_index', [
    (GCNConv(in_channels, 64), 'x, edge_index -> x'),
    ReLU(inplace=True),
    (GCNConv(64, 64), 'x, edge_index -> x'),
    ReLU(inplace=True),
    Linear(64, out_channels),
])
"""
def global_mean_pool(x: Tensor, batch: Optional[Tensor],
                     size: Optional[int] = None) -> Tensor:
    r"""Returns batch-wise graph-level-outputs by averaging node features
    across the node dimension, so that for a single graph
    :math:`\mathcal{G}_i` its output is computed by

    .. math::
        \mathbf{r}_i = \frac{1}{N_i} \sum_{n=1}^{N_i} \mathbf{x}_n

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example.
        size (int, optional): Batch-size :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)
    """
    if batch is None:
        return x.mean(dim=0, keepdim=True)
    size = int(batch.max().item() + 1) if size is None else size
    return scatter(x, batch, dim=0, dim_size=size, reduce='mean')


class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 120,
        last_layer_dim_vf: int = 120,
    ):
        super(CustomNetwork, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        self.nodes = last_layer_dim_pi

        #how many 'jumps' / for loops
        self.iterations = 1

        self.Meta_Layer_p1 = MetaLayer(
            Edge_Model(
                input_features=17, hidden_features = 128, output_features = 2),
            Node_Model(
                input_features_mlp1=7, hidden_features_mlp1=128, output_features_mlp1=128,
                input_features_mlp2=134, hidden_features_mlp2=256, output_features_mlp2=10),
            Global_Model(
                input_features = 16, hidden_features = 128, output_features = 12)
            )

        self.Meta_Layer_p2 = MetaLayer(
            Edge_Model(
                input_features=34, hidden_features = 128, output_features = 1),
            Node_Model(
                input_features_mlp1=11, hidden_features_mlp1=128, output_features_mlp1=128,
                input_features_mlp2=139, hidden_features_mlp2=256, output_features_mlp2=1),
            Global_Model(
                input_features = 13, hidden_features = 128, output_features = 6)
            )

        #MetaLayer Value Net
        self.Meta_Layer_v1 = MetaLayer(
            Edge_Model(
                input_features=17, hidden_features = 128, output_features = 2),
            Node_Model(
                input_features_mlp1=7, hidden_features_mlp1=128, output_features_mlp1=128,
                input_features_mlp2=134, hidden_features_mlp2=256, output_features_mlp2=10),
            Global_Model(
                input_features = 16, hidden_features = 128, output_features = 12)
            )

        self.Meta_Layer_v2 = MetaLayer(
            Edge_Model(
                input_features=34, hidden_features = 128, output_features = 1),
            Node_Model(
                input_features_mlp1=11, hidden_features_mlp1=128, output_features_mlp1=128,
                input_features_mlp2=139, hidden_features_mlp2=256, output_features_mlp2=1),
            Global_Model(
                input_features = 13, hidden_features = 128, output_features = 6)
            )

        self.episode = 0

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        batch_size = features.shape[0]
        #make tensors
        
        #print(features)
        degrees = features[:,:self.nodes]
        #print(degrees)
        capacities = features[:,self.nodes:self.nodes*2]
        #print(capacities.shape)
        incoming_transmissions = features[:,self.nodes*2:self.nodes*3]
        #print(incoming_transmissions.shape)
        outgoing_transmissions = features[:,self.nodes*3:self.nodes*4]
        #print(outgoing_transmissions.shape)
        total_transmissions = features[:,self.nodes*4:self.nodes*5]
        #print(total_transmissions.shape)
        graph_features = features[:,self.nodes*5:self.nodes*5+6]
        #print(graph_features.shape)
        no_edgeattributes = int((features.shape[1]-(5*self.nodes+6))/3)
        #print(no_edgeattributes)
        edge_attributes = features[:,self.nodes*5+6:self.nodes*5+6+no_edgeattributes]
        #print(source_nodes.shape)
        source_nodes = features[:,self.nodes*5+6+no_edgeattributes:self.nodes*5+6+2*no_edgeattributes]
        #print(destination_nodes.shape)
        destination_nodes = features[:,self.nodes*5+6+2*no_edgeattributes:self.nodes*5+6+3*no_edgeattributes]
        #print(edge_attributes.shape)
        
        if batch_size==1:
            if self.episode==0:
                self.source_nodes = torch.tensor([])
                self.destination_nodes = torch.tensor([])
            #print(degrees.shape)
            source_nodes = source_nodes
            destination_nodes = destination_nodes
            self.source_nodes = torch.cat((self.source_nodes, torch.Tensor(source_nodes+self.episode)))
            self.destination_nodes = torch.cat((self.destination_nodes, torch.Tensor(destination_nodes+self.episode)))
            #print(source_nodes)
            #print(destination_nodes)
            source_nodes = source_nodes.flatten()
            destination_nodes = destination_nodes.flatten()
            self.episode += self.nodes # +120/1040/60k
        else:
            self.source_nodes = self.source_nodes.flatten()
            self.destination_nodes = self.destination_nodes.flatten()
            source_nodes = self.source_nodes
            destination_nodes = self.destination_nodes
            self.episode = 0
            
        edge_attributes = edge_attributes.flatten()

        edge_index = torch.stack((source_nodes, destination_nodes), dim=0)
        edge_index = edge_index.long()
        #print(edge_index.shape)
        #formatting for MetaLayer

        edge_attributes = torch.reshape(edge_attributes, (edge_attributes.shape[0], 1))
        #print(edge_attributes.shape)

        incoming_transmissions_ml = torch.reshape(incoming_transmissions, (self.nodes*batch_size, 1))
        outgoing_transmissions_ml = torch.reshape(outgoing_transmissions, (self.nodes*batch_size, 1))
        total_transmissions_ml = torch.reshape(total_transmissions, (self.nodes*batch_size, 1))
        capacities_ml = torch.reshape(capacities, (self.nodes*batch_size, 1))
        degrees_ml = torch.reshape(degrees, (self.nodes*batch_size, 1))
        
        #print(graph_features.shape)
        #print(graph_features.contiguous().view(-1,1))
        node_features = torch.cat((capacities_ml, degrees_ml, incoming_transmissions_ml, outgoing_transmissions_ml, total_transmissions_ml), dim = 1)
        #print(node_features.shape)
        #print(graph_features.shape)
        """MetaLayer"""
        """recurrent ML Layer"""
        for i in range(self.iterations):
            h1_nodes_policy, h1_edges_policy, h1_graph_policy = self.Meta_Layer_p1(
                x = node_features, edge_attr = edge_attributes, u = graph_features, edge_index = edge_index, batch= batch_size)
    
        h2_nodes_policy, h2_edges_policy, h2_graph_policy = self.Meta_Layer_p2(
            x = h1_nodes_policy, edge_attr = h1_edges_policy, u = h1_graph_policy, edge_index = edge_index, batch = batch_size)
        
        updated_nodes_policy, updated_edges_policy, updated_graph_policy = h2_nodes_policy, h2_edges_policy, h2_graph_policy
        
        """value"""
        """recurrent ML Layer"""
        for i in range(self.iterations):
            h1_nodes_value, h1_edges_value, h1_graph_value = self.Meta_Layer_v1(
                x = node_features, edge_attr = edge_attributes, u = graph_features, edge_index = edge_index, batch= batch_size)
  
        h2_nodes_value, h2_edges_value, h2_graph_value = self.Meta_Layer_v2(
            x = h1_nodes_value, edge_attr = h1_edges_value, u = h1_graph_value, edge_index = edge_index, batch = batch_size)
        
        updated_nodes_value, updated_edges_value, updated_graph_value = h2_nodes_value, h2_edges_value, h2_graph_value
        
        #formatting MetaLayer
        policy = torch.transpose(updated_nodes_policy, 0, 1)
        value = torch.transpose(updated_nodes_value, 0, 1)

        policy = policy.view(batch_size, self.nodes)
        value = value.view(batch_size, self.nodes)

        return policy, value

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        batch_size = features.shape[0]
        #make tensors

        degrees = features[:,:self.nodes]
        #print(degrees.shape)
        capacities = features[:,self.nodes:self.nodes*2]
        #print(capacities.shape)
        incoming_transmissions = features[:,self.nodes*2:self.nodes*3]
        #print(incoming_transmissions.shape)
        outgoing_transmissions = features[:,self.nodes*3:self.nodes*4]
        #print(outgoing_transmissions.shape)
        total_transmissions = features[:,self.nodes*4:self.nodes*5]
        #print(total_transmissions.shape)
        graph_features = features[:,self.nodes*5:self.nodes*5+6]
        #print(graph_features.shape)
        no_edgeattributes = int((features.shape[1]-(5*self.nodes+6))/3)
        #print(no_edgeattributes)
        edge_attributes = features[:,self.nodes*5+6:self.nodes*5+6+no_edgeattributes]
        #print(source_nodes.shape)
        source_nodes = features[:,self.nodes*5+6+no_edgeattributes:self.nodes*5+6+2*no_edgeattributes]
        #print(destination_nodes.shape)
        destination_nodes = features[:,self.nodes*5+6+2*no_edgeattributes:self.nodes*5+6+3*no_edgeattributes]
        #print(edge_attributes.shape)

        if batch_size==1:
            #print(degrees.shape)
            source_nodes = source_nodes
            destination_nodes = destination_nodes
            self.source_nodes = torch.cat((self.source_nodes, torch.Tensor(source_nodes+self.episode)))
            self.destination_nodes = torch.cat((self.destination_nodes, torch.Tensor(destination_nodes+self.episode)))
            #print(source_nodes)
            #print(destination_nodes)
            source_nodes = source_nodes.flatten()
            destination_nodes = destination_nodes.flatten()
            self.episode += self.nodes
        else:
            self.source_nodes = self.source_nodes.flatten()
            self.destination_nodes = self.destination_nodes.flatten()
            source_nodes = self.source_nodes
            destination_nodes = self.destination_nodes
            self.episode = 0

        edge_attributes = edge_attributes.flatten()

        edge_index = torch.stack((source_nodes, destination_nodes), dim=0)
        edge_index = edge_index.long()
        #print(edge_index.shape)
        #formatting for MetaLayer
        edge_attributes = torch.reshape(edge_attributes, (edge_attributes.shape[0], 1))
        #print(edge_attributes.shape)

        incoming_transmissions_ml = torch.reshape(incoming_transmissions, (self.nodes*batch_size, 1))
        outgoing_transmissions_ml = torch.reshape(outgoing_transmissions, (self.nodes*batch_size, 1))
        total_transmissions_ml = torch.reshape(total_transmissions, (self.nodes*batch_size, 1))
        capacities_ml = torch.reshape(capacities, (self.nodes*batch_size, 1))
        degrees_ml = torch.reshape(degrees, (self.nodes*batch_size, 1))
        
        #print(graph_features.shape)
        #print(graph_features.contiguous().view(-1,1))
        node_features = torch.cat((capacities_ml, degrees_ml, incoming_transmissions_ml, outgoing_transmissions_ml, total_transmissions_ml), dim = 1)
        
        #print(graph_features.shape)
        """MetaLayer"""
        """recurrent ML Layer"""
        for i in range(self.iterations):
            h1_nodes_policy, h1_edges_policy, h1_graph_policy = self.Meta_Layer_p1(
                x = node_features, edge_attr = edge_attributes, u = graph_features, edge_index = edge_index, batch= batch_size)
    
        h2_nodes_policy, h2_edges_policy, h2_graph_policy = self.Meta_Layer_p2(
            x = h1_nodes_policy, edge_attr = h1_edges_policy, u = h1_graph_policy, edge_index = edge_index, batch = batch_size)
        
        updated_nodes_policy, updated_edges_policy, updated_graph_policy = h2_nodes_policy, h2_edges_policy, h2_graph_policy
        

        #formatting MetaLayer
        policy = torch.transpose(updated_nodes_policy, 0, 1)

        policy = policy.view(batch_size, self.nodes)
        
        return policy

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        batch_size = features.shape[0]
        #make tensors
        degrees = features[:,:self.nodes]
        #print(degrees.shape)
        capacities = features[:,self.nodes:self.nodes*2]
        #print(capacities.shape)
        incoming_transmissions = features[:,self.nodes*2:self.nodes*3]
        #print(incoming_transmissions.shape)
        outgoing_transmissions = features[:,self.nodes*3:self.nodes*4]
        #print(outgoing_transmissions.shape)
        total_transmissions = features[:,self.nodes*4:self.nodes*5]
        #print(total_transmissions.shape)
        graph_features = features[:,self.nodes*5:self.nodes*5+6]
        #print(graph_features.shape)
        no_edgeattributes = int((features.shape[1]-(5*self.nodes+6))/3)
        #print(no_edgeattributes)
        edge_attributes = features[:,self.nodes*5+6:self.nodes*5+6+no_edgeattributes]
        #print(source_nodes.shape)
        source_nodes = features[:,self.nodes*5+6+no_edgeattributes:self.nodes*5+6+2*no_edgeattributes]
        #print(destination_nodes.shape)
        destination_nodes = features[:,self.nodes*5+6+2*no_edgeattributes:self.nodes*5+6+3*no_edgeattributes]
        #print(edge_attributes.shape)

        edge_attributes = edge_attributes.flatten()
        source_nodes = source_nodes.flatten()
        destination_nodes = destination_nodes.flatten()

        edge_index = torch.stack((source_nodes, destination_nodes), dim=0)
        edge_index = edge_index.long()
        #print(edge_index.shape)
        #formatting for MetaLayer
        edge_attributes = torch.reshape(edge_attributes, (edge_attributes.shape[0], 1))

        #print(edge_attributes.shape)

        incoming_transmissions_ml = torch.reshape(incoming_transmissions, (self.nodes*batch_size, 1))
        outgoing_transmissions_ml = torch.reshape(outgoing_transmissions, (self.nodes*batch_size, 1))
        total_transmissions_ml = torch.reshape(total_transmissions, (self.nodes*batch_size, 1))
        capacities_ml = torch.reshape(capacities, (self.nodes*batch_size, 1))
        degrees_ml = torch.reshape(degrees, (self.nodes*batch_size, 1))
        
        node_features = torch.cat((capacities_ml, degrees_ml, incoming_transmissions_ml, outgoing_transmissions_ml, total_transmissions_ml), dim = 1)
        
        """value"""
        """recurrent ML Layer"""
        for i in range(self.iterations):
            h1_nodes_value, h1_edges_value, h1_graph_value = self.Meta_Layer_v1(
                x = node_features, edge_attr = edge_attributes, u = graph_features, edge_index = edge_index, batch= batch_size)
  
        h2_nodes_value, h2_edges_value, h2_graph_value = self.Meta_Layer_v2(
            x = h1_nodes_value, edge_attr = h1_edges_value, u = h1_graph_value, edge_index = edge_index, batch = batch_size)
        
        updated_nodes_value, updated_edges_value, updated_graph_value = h2_nodes_value, h2_edges_value, h2_graph_value
        
        #formatting MetaLayer
        value = torch.transpose(updated_nodes_value, 0, 1)
        value = value.view(batch_size, self.nodes)
        
        return value


class CustomActorCriticPolicy_120_ML(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
    ):

        super(CustomActorCriticPolicy_120_ML, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim)
