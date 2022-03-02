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


<<<<<<< HEAD
<<<<<<< HEAD
train_input_data = "/Users/admin/pyoptes/src/inputs_waxman_120_sent_sci2.csv"
train_targets_data = "/Users/admin/pyoptes/src/targets_waxman_120_sent_sci2.csv"


x, y = process.postprocessing(train_input_data, train_targets_data, split = 20000, grads = True)

data_list = prep_conv(x,y)

print(len(data_list))

loader = DataLoader(data_list, batch_size = 256, shuffle = True)

model = Net(16, dataset=data_list[0]).double() # gdc = gdc).double()

#torch.load(model.state_dict(), "/Users/admin/pyoptes/src/gat_wax_120.pth")

=======
=======
>>>>>>> 7d652ef (commit)
train_input_data = "/Users/admin/pyoptes/src/inputs_waxman_120.csv"
train_targets_data = "/Users/admin/pyoptes/src/targets_waxman_120.csv"


x, y = process.postprocessing(train_input_data, train_targets_data, split = 1000, grads = True)

data_list = prep_conv(x,y)

loader = DataLoader(data_list, batch_size = 1, shuffle = True)

model = Net(16, dataset=data_list[0]).double() # gdc = gdc).double()

<<<<<<< HEAD
>>>>>>> 7d652ef (commit)
=======
>>>>>>> 7d652ef (commit)
epochs = 500
criterion = nn.L1Loss() 

optimizer_params = {"lr": 0.001, "weight_decay": 0.005, "betas": (0.9, 0.999)}
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

    for batch in loader:
      
      optimizer.zero_grad()
      targets = batch.y.unsqueeze(-1) #= [32,1]
      output = model(batch)

      #print(targets.shape, output.shape)

      loss = criterion(output, targets)
      loss.backward()
      optimizer.step()
      train_loss.append(loss.item())

      for j, val in enumerate(output):
        true.append(targets[j].item())
        pred.append(output[j].item())

    acc = explained_variance_score(true, pred)
    return np.mean(train_loss), acc

<<<<<<< HEAD
<<<<<<< HEAD
def validate(loader, model):
    model.eval()
    true = []
    pred = []
    val_loss = []

    for batch in loader:
      
      targets = batch.y.unsqueeze(-1) #= [32,1]
      output = model(batch)

      #print(targets.shape, output.shape)

      loss = criterion(output, targets)
      val_loss.append(loss.item())

      for j, val in enumerate(output):
        true.append(targets[j].item())
        pred.append(output[j].item())

    acc = explained_variance_score(true, pred)
    return np.mean(val_loss), acc

val_loss = []
val_acc = []

total_loss = []
total_acc = []

val_loss_prev = np.inf
=======
total_loss = []
total_acc = []
train_loss_prev = np.inf
>>>>>>> 7d652ef (commit)
=======
total_loss = []
total_acc = []
train_loss_prev = np.inf
>>>>>>> 7d652ef (commit)
for epoch in range(epochs):
  train_loss, train_acc = training(loader, model, criterion, optimizer) 
  total_loss.append(train_loss)
  total_acc.append(train_acc)
<<<<<<< HEAD
<<<<<<< HEAD


  valloss, valacc = validate(loader, model)
  val_loss.append(valloss)
  val_acc.append(valacc)

  if valloss < val_loss_prev:
    train_loss_prev = train_loss
    print(f'epoch: {epoch+1}, train loss: {train_loss_prev}, train acc: {train_acc}, val loss: {valloss}, val acc: {valacc}')
    torch.save(model.state_dict(), "/Users/admin/pyoptes/src/gat_wax_120.pth")


plt.figure()
plt.plot(np.arange(epochs), total_loss, label = "training loss")
plt.plot(np.arange(epochs), val_loss, label = "validation loss")
=======
=======
>>>>>>> 7d652ef (commit)
  if train_loss < train_loss_prev:
    train_loss_prev = train_loss
    print(f'epoch: {epoch+1}, train loss: {train_loss_prev}, train acc: {train_acc}')

plt.figure()
plt.plot(np.arange(epochs), total_loss, label = "training loss")
<<<<<<< HEAD
>>>>>>> 7d652ef (commit)
=======
>>>>>>> 7d652ef (commit)
plt.xlabel("epochs")
plt.ylabel("loss")

plt.figure()
plt.plot(np.arange(epochs), total_acc, label = "training acc")
<<<<<<< HEAD
<<<<<<< HEAD
plt.plot(np.arange(epochs), val_acc, label = "validation acc")
=======
>>>>>>> 7d652ef (commit)
=======
>>>>>>> 7d652ef (commit)
plt.xlabel("epochs")
plt.ylabel("accuracy")

plt.show()