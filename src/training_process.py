import os
from xml.dom import HierarchyRequestErr
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import random
from tqdm import tqdm
from operator import xor
from scipy.stats import gaussian_kde as kde
import pylab as plt
from pyoptes import set_seed
from pyoptes.optimization.budget_allocation import target_function as f
import csv
import warnings
import networkx as nx
from pyoptes.optimization.budget_allocation.supervised_learning.utils import Loader as Loader
from pyoptes.optimization.budget_allocation.supervised_learning.utils import processing as process
from pyoptes.optimization.budget_allocation.supervised_learning.utils import model_selection as model_selection
from pyoptes.optimization.budget_allocation.supervised_learning.utils import device as get_device
from pyoptes.optimization.budget_allocation.supervised_learning.utils import training_process as train_nn
from torch.utils.tensorboard import SummaryWriter
from ray import tune

set_seed(1)

#writer = SummaryWriter(log_dir = "/Users/admin/pyoptes/src")

device = get_device()

#load inputs and targets
#inputs = "/Users/admin/pyoptes/src/pyoptes/optimization/budget_allocation/supervised_learning/training_data/ba_inputs_100k.csv"
#targets = "/Users/admin/pyoptes/src/pyoptes/optimization/budget_allocation/supervised_learning/training_data/ba_targets_100k.csv"

inputs = "/Users/admin/pyoptes/src/pyoptes/optimization/budget_allocation/supervised_learning/training_data/wx_inputs.csv"
targets = "/Users/admin/pyoptes/src/pyoptes/optimization/budget_allocation/supervised_learning/training_data/wx_targets.csv"

#split data into training and test data
train_data, test_data = process.postprocessing(inputs, targets, split = 1, grads = False)

trainset = DataLoader(train_data, batch_size = 1, shuffle=True)
testset = DataLoader(test_data, batch_size = 128, shuffle=True)


"""define some hyperparameters"""
nodes = 120
output_dimensions = 120
epochs = 500
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
network = "Waxman"
layer_dimensions = {'small': (128, 64, 32, 16, output_dimensions), 'medium' : (256, 128, 64, 32, output_dimensions), 'big' : (512, 256, 128, 64, output_dimensions)}
pick = "FCN" #FCN or RNN
model = model_selection.set_model(pick, dim = nodes, layer_dimensions = layer_dimensions["small"])
model.to(device)

#load pre-trained model
model_state = f"/Users/admin/pyoptes/src/pyoptes/optimization/budget_allocation/supervised_learning/trained_nn_states/{network}_120_{pick}_per_node.pth"
#model.load_state_dict(torch.load(model_state))

#criterion = nn.MSELoss() 
criterion = nn.L1Loss() #mean absolut error
#learning_rate = 3.5

lr = -3

"""opt_params for Adam/W"""
optimizer_params = {"lr": 10**lr, "weight_decay": 0.001, "betas": (0.9, 0.999)}
"""opt_params for SGD"""
#optimizer_params = {"lr": 0.02, "weight_decay": 0.0005, "momentum": True}
"""opt_params for Adagrad"""
#optimizer_params = {"lr": 0.01}
"""Opt AdamW"""
optimizer = optim.AdamW(model.parameters(), **optimizer_params)
"""Opt Adam performs like AdamW"""
#optimizer = optim.Adam(model.parameters(), **optimizer_params)
"""This algorithm performs best for sparse data because it decreases the learning rate faster for frequent parameters, 
and slower for parameters infrequent parameter."""
#optimizer = optim.Adagrad(model.parameters(), **optimizer_params)

"""training process"""
#variables for later storing
plotter_train_loss = []
plotter_test_loss = []
plotter_train_acc = []
plotter_test_acc = []

#boundary
val_loss_init = np.inf

for epoch in range(1, epochs + 1):

    train_loss, train_acc = train_nn.train(trainloader = trainset, model=model, device=device, optimizer = optimizer,
                                  criterion=criterion, verbose=50)
    val_loss, val_acc = train_nn.validate(valloader= testset, model=model, device=device, criterion=criterion, verbose=10)
    
    #writer.add_scalar(f'Loss/test {pick} {nodes} nodes {network}', val_loss, epoch)
    ##writer.add_scalar(f'Accuracy/test {pick} {nodes} nodes {network}', val_acc, epoch)

    plotter_train_loss.append(train_loss)
    plotter_test_loss.append(val_loss)

    plotter_train_acc.append(train_acc)
    plotter_test_acc.append(val_acc)

    if val_loss < val_loss_init:
      print(f"epoch {epoch}:, train loss: {train_loss:.4f}, train acc: {train_acc:.4f}, validation loss: {val_loss:.4f}, validation acc: {val_acc:.4f}")
      torch.save(model.state_dict(), model_state)
      val_loss_init = val_loss

plt_train_acc = np.array(plotter_train_acc)*100
plt_test_acc = np.array(plotter_test_acc)*100


torch.save(model.state_dict(), model_state)

plt.figure(figsize=(5,5))
plt.plot(np.arange(epochs), plotter_train_loss, label = "training loss")
plt.plot(np.arange(epochs), plotter_test_loss, label = "test loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title(f"{pick}")
plt.legend()

plt.figure(figsize=(5,5))
plt.axis([0, epochs, 0, 100])
plt.plot(np.arange(epochs), plt_train_acc, label = "Training Accuracy")
plt.plot(np.arange(epochs), plt_test_acc, label = "Test Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title(f"{pick}")
plt.legend()
plt.show()
