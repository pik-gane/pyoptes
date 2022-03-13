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
from pyoptes.optimization.budget_allocation.supervised_learning.utils import Loader as Loader, device
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
from Graph_DataList import prepare_convolutions as prep_conv
from pyoptes.optimization.budget_allocation.supervised_learning.utils import device as get_device

train_input_data = "/Users/admin/pyoptes/src/pyoptes/optimization/budget_allocation/supervised_learning/training_data/wx_inputs.csv"
train_targets_data = "/Users/admin/pyoptes/src/pyoptes/optimization/budget_allocation/supervised_learning/training_data/wx_targets.csv"

x, y = process.postprocessing(train_input_data, train_targets_data, split = 20000, grads = True)

device = get_device()

data_list = prep_conv(x,y)

print(len(data_list))

loader = DataLoader(data_list, batch_size = 128, shuffle = True)
model = Net(16, dataset=data_list[0]).double() # gdc = gdc).double()

epochs = 50
criterion = nn.L1Loss() 

lr = -2

optimizer_params = {"lr": 10**lr, "weight_decay": 0.005, "betas": (0.9, 0.999)}
#Ridge regression; it is a technique where the sum of squared parameters, 
# or weights of a model (multiplied by some coefficient) is added into the loss function as a penalty 
# term to be minimized.

optimizer = optim.AdamW(model.parameters(), **optimizer_params)

#optimizer_params = {"lr": 0.001, "weight_decay": 0.005}# "betas": (0.9, 0.999)}
#optimizer = optim.Adam(model.parameters(), **optimizer_params)

#optimizer_params = {"lr": 0.1, "weight_decay": 0.0005, "momentum": True}
#optimizer = optim.SGD(model.parameters(), **optimizer_params)

#optimizer = optim.Adagrad(model.parameters(), lr =  0.1)

def training(loader, model, criterion, optimizer):
    model.train()
    true = []
    pred = []
    train_loss = []
    acc = []

    for batch in loader:
      #batch.y.to(device)
      #batch.to(device)

      optimizer.zero_grad()
      targets = batch.y.unsqueeze(-1) #= [32,1]
      output = model(batch)

      #print(targets.shape, output.shape)
      loss = criterion(output, targets)
      loss.backward()
      optimizer.step()
      train_loss.append(loss.item())
      #print(output[0])
      #print(targets[0])
      acc.append(explained_variance_score(targets.detach(), output.detach()))
      #print(acc)
      #for j, val in enumerate(output):
              #true.append(targets[j].detach())
              #pred.append(output[j].detach())
      #acc.append(explained_variance_score(targets[j].detach(), output[j].detach()))
      #acc = explained_variance_score(true, pred) #1 - var(y-y_hat)/var(y) 
    return np.mean(train_loss), np.mean(acc)

def validate(loader, model):
    model.eval()
    true = []
    pred = []
    val_loss = []
    acc = []

    for batch in loader:
      #batch.to(device)
      #batch.y.to(device)

      targets = batch.y.unsqueeze(-1) #= [32,1]
      output = model(batch)
      #print(targets.shape, output.shape)
      loss = criterion(output, targets)
      val_loss.append(loss.item())
      acc.append(explained_variance_score(targets.detach(), output.detach()))

    #for j, val in enumerate(output):
    #true.append(targets[j].detach())
    #pred.append(output[j].detach())
    # acc.append(explained_variance_score(targets[j].detach(), output[j].detach()))
    #acc = explained_variance_score(true, pred) #1 - var(y-y_hat)/var(y) 

    return np.mean(val_loss), np.mean(acc)
     
val_loss = []
val_acc = []

train_loss_prev = np.inf
total_loss = []
total_acc = []

val_loss_prev = np.inf

for epoch in range(epochs):
  train_loss, train_acc = training(loader, model, criterion, optimizer) 
  total_loss.append(train_loss)
  total_acc.append(train_acc)
  valloss, valacc = validate(loader, model)
  val_loss.append(valloss)
  val_acc.append(valacc)

  if valloss < val_loss_prev:
  #if train_loss < train_loss_prev:
    train_loss_prev = train_loss
    print(f'epoch: {epoch+1}, train loss: {train_loss_prev}, train acc: {train_acc}, val loss: {valloss}, val acc: {valacc}')
    torch.save(model.state_dict(), "/Users/admin/pyoptes/src/gat_wax_120.pth")


plt.figure()
plt.plot(np.arange(epochs), total_loss, label = "training loss")
plt.plot(np.arange(epochs), val_loss, label = "validation loss")
plt.xlabel("epochs")
plt.ylabel("loss")

plt.figure()
plt.plot(np.arange(epochs), total_acc, label = "training acc")
plt.plot(np.arange(epochs), val_acc, label = "validation acc")
plt.xlabel("epochs")
plt.ylabel("accuracy")

plt.show()