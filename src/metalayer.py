from torch.nn import Sequential
from torch import nn
from torch_scatter import scatter_mean
from torch_geometric.nn import MetaLayer
import torch
from tkinter import Y
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
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from gnn import Net
from gnn import get_features
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch import optim
from sklearn.metrics import explained_variance_score, mean_squared_error
from prepare_conv import prepare_convolutions as prep_conv
from functools import partial
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from tqdm import tqdm

train_input_data = "/Users/admin/pyoptes/src/inputs_waxman_120_sent_sci2.csv"
train_targets_data = "/Users/admin/pyoptes/src/targets_waxman_120_sent_sci2.csv"

x, y = process.postprocessing(train_input_data, train_targets_data, split = 20000, grads = True)

data_list = prep_conv(x,y)
train_loader = DataLoader(data_list[5000:], batch_size = 128, shuffle = True)
test_loader = DataLoader(data_list[:5000], batch_size = 128, shuffle = True)

class Edge_Model(torch.nn.Module):

    def __init__(self, ins, hiddens, outs):
        super(Edge_Model, self).__init__()
        #print(in_channels)
        self.edge_mlp = Sequential(
            nn.Linear(ins+5, hiddens),
            nn.ReLU(),
            nn.Linear(hiddens, hiddens),
            nn.ReLU(),
            nn.Linear(hiddens, ins))

    def forward(self, src, dst, edge_attr, u, batch):  #source attribute, destination attribute,edge features, 
        u = u.view(-1, 1)   
        edge_attr = edge_attr.view(-1, 1) 
        #(u.shape, edge_attr.shape)
        out = torch.cat([src, dst, edge_attr, u[batch]], 1)
        out = self.edge_mlp(out)
        #if self.residuals:
        #    out = out + edge_attr
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
        src, dest = edge_index
        
        out = torch.cat([x[src], edge_attr], dim=1) #updated edge // stack all edge features 
        
        #print(out.shape, x[src].shape, x[dest].shape)

        out = self.node_mlp_1(out) #updated feature of our edges

        out = scatter_mean(out, dest, dim=0, dim_size=x.size(0)) #new target feature 

        u = u.view(-1,1)
        out = torch.cat([x, out, u[batch]], dim=1) 
        
        out = self.node_mlp_2(out) #updated feature of target node
        #if self.residuals:
        #    out = out + edge_attr
        return out  

#src = source, dest = target, node_model = target node model, 

class Global_Model(torch.nn.Module):

    def __init__(self, ins, hiddens, outs):
        super(Global_Model, self).__init__()
        #print(in_channels)
        #print(outs)
        self.global_mlp = Sequential(
            nn.Linear(ins+2, hiddens),
            nn.ReLU(),
            nn.Linear(hiddens, hiddens),
            nn.ReLU(),
            nn.Linear(hiddens, outs))
    def forward(self, x, edge_index, edge_attr, u, batch):
        u = u.view(-1, 6)
        #print(u.shape)
        x = scatter_mean(x, batch, dim=0)
        #print(x.shape)
        out = torch.cat([u, x], dim=1)
        out = self.global_mlp(out)

        #if self.residuals:
        #    out = out + edge_attr

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

        #self.layer_5 = MetaLayer(Global_Model(ins_graphs, hiddens, outs=outs-5))

    # Defining the forward pass    
    def forward(self, x, edge_attr, u, edge_index, batch):
        
        x, edge_attr, u = self.layer_1(x=x, edge_attr=edge_attr, edge_index=edge_index, u=u, batch=batch)
        x, edge_attr, u = self.layer_2(x=x, edge_attr=edge_attr, edge_index=edge_index, u=u, batch=batch)
        x, edge_attr, u = self.layer_3(x=x, edge_attr=edge_attr, edge_index=edge_index, u=u, batch=batch)
        x, edge_attr, u = self.layer_4(x=x, edge_attr=edge_attr, edge_index=edge_index, u=u, batch=batch)

        #print(x.shape, edge_attr.shape, u.shape)
        #x, edge_attr, u = self.layer_5(x=x, edge_attr=edge_attr, edge_index=edge_index, u=u, batch=batch)
        #print(x.shape, edge_attr.shape, u.shape)
        return u

model = meta_layer(ins_nodes = 2, ins_edges = 1, ins_graphs = 6, hiddens= 16, outs = 6).double() # gdc = gdc).double()
epochs = 10000
criterion = nn.L1Loss() 

optimizer_params = {"lr": 0.001, "weight_decay": 0.005, "betas": (0.9, 0.999)}
optimizer = optim.AdamW(model.parameters(), **optimizer_params)


#model.load_state_dict(torch.load("/Users/admin/pyoptes/src/meta_layer.pth"))

#optimizer_params = {"lr": 0.1, "weight_decay": 0.0005}
#optimizer = optim.Adam(model.parameters(), **optimizer_params)

#optimizer_params = {"lr": 0.1}
#optimizer = optim.Adagrad(model.parameters(), **optimizer_params)

def training(loader, model, criterion, optimizer):
    model.train()
    true = []
    pred = []
    train_loss = []
    for batch in loader:
        optimizer.zero_grad()
        targets = batch.y.unsqueeze(-1) #= [32,1]

        x, edge_index, edge_weight, u, batch = batch.x, batch.edge_index, batch.weight, batch.edge_attr, batch.batch
        
        u = model.forward(x = x, edge_attr = edge_weight, u = u, edge_index = edge_index, batch = batch)
        
        #print(u.shape)

        loss = criterion(u, targets)
        loss.backward()
        optimizer.step()

        #print(u[0].item(), targets[0].item())

        train_loss.append(loss.item())

        for j, val in enumerate(u):
            true.append(targets[j].item())
            pred.append(u[j].item())

    acc = explained_variance_score(true, pred)
    return np.mean(train_loss), acc


def validate(valloader: DataLoader, model: torchvision.models):
    
    model.eval()
    true = []
    pred = []
    val_loss = []
    with torch.no_grad():

        for batch in valloader:
            targets = batch.y.unsqueeze(-1) #= [32,1]

            x, edge_index, edge_weight, u, batch = batch.x, batch.edge_index, batch.weight, batch.edge_attr, batch.batch
            
            u = model.forward(x = x, edge_attr = edge_weight, u = u, edge_index = edge_index, batch = batch)
            
            loss = criterion(u, targets)
        
            #print(u[0].item(), targets[0].item())

            val_loss.append(loss.item())

            for j, val in enumerate(u):
                true.append(targets[j].item())
                pred.append(u[j].item())

    acc = explained_variance_score(true, pred)
    return np.mean(val_loss), acc


total_loss = []
total_acc = []

_val_loss = []
_val_acc = []

train_loss_prev = np.inf
val_loss_prev = np.inf

for epoch in range(epochs):

  train_loss, train_acc = training(train_loader, model, criterion, optimizer) 
  total_loss.append(train_loss)
  total_acc.append(train_acc)
  
  val_loss, val_acc = validate(test_loader, model) 
  _val_loss.append(val_loss)
  _val_acc.append(val_acc)


  if train_loss < train_loss_prev:
    train_loss_prev = train_loss
    val_loss_prev = val_loss
    torch.save(model.state_dict(), "/Users/admin/pyoptes/src/meta_layer20T.pth")
    print(f'epoch: {epoch+1}, train loss: {train_loss_prev}, train acc: {train_acc}, val loss: {val_loss_prev}, val acc: {val_acc}')

plt.figure()
plt.plot(np.arange(epochs), np.sqrt(total_loss), label = "training loss")
plt.plot(np.arange(epochs), np.sqrt(_val_loss), label = "val loss")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend()

plt.figure()
plt.plot(np.arange(epochs), total_acc, label = "training acc")
plt.plot(np.arange(epochs), _val_acc, label = "val acc")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()

plt.show()