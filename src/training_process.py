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

writer = SummaryWriter(log_dir = "/Users/admin/pyoptes/src")

device = get_device()

train_input_data = "/Users/admin/pyoptes/src/inputs_waxman_120_sent_sci2.csv"
train_targets_data = "/Users/admin/pyoptes/src/targets_waxman_120_sent_sci2.csv"

model_state = "/Users/admin/pyoptes/src/waxman_120_sci2.pth"

#train_input_data = "/Users/admin/pyoptes/src/input_data_waxman_fast.csv"
#train_targets_data = "/Users/admin/pyoptes/src/label_data_waxman_fast.csv"

#test_input_data = "/Users/admin/pyoptes/src/pyoptes/optimization/budget_allocation/supervised_learning/input_data_round.csv"
#test_targets_data = "/Users/admin/pyoptes/src/pyoptes/optimization/budget_allocation/supervised_learning/targets_data_round.csv"

#print(f'\n\nSize of training inputs, targets: {len(train_data)} \n\nSize of test inputs, targets: {len(val_input)}\n\n')

train_data, test_data = process.postprocessing(train_input_data, train_targets_data, split = 5000, grads = False)

inputs_train_data = DataLoader(train_data, batch_size = 128, shuffle=True)
targets_test_data = DataLoader(test_data, batch_size = 128, shuffle=True)


random.seed(10)

epochs = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#network = "Waxman" #"lattice" #"Barabasi-Albert"

network = "Barabasi-Albert"
hidden_dims = (128, 64, 32, 16)
nodes = 120
pick = "RNN"

model = model_selection.set_model(pick, dim = nodes, hidden_dims = hidden_dims)
model.to(device)

#criterion = nn.MSELoss() 
criterion = nn.L1Loss() #mean absolut error
#learning_rate = 3.5
"""opt_params for Adam/W"""
optimizer_params = {"lr": 0.001, "weight_decay": 0.01, "betas": (0.9, 0.999)}
"""opt_params for SGD"""
#optimizer_params = {"lr": 0.02, "weight_decay": 0.0005, "momentum": True}
"""opt_params for Adagrad"""
#optimizer_params = {"lr": 0.01}
"""Opt AdamW"""
optimizer = optim.AdamW(model.parameters(), **optimizer_params)
"""Opt Adam performs like AdamW"""
#optimizer = optim.Adam(model.parameters(), **optimizer_params)
"""SGD not working"""
#optimizer = optim.SGD(model.parameters(), **optimizer_params)
"""This algorithm performs best for sparse data because it decreases the learning rate faster for frequent parameters, 
and slower for parameters infrequent parameter."""
#optimizer = optim.Adagrad(model.parameters(), **optimizer_params)


plotter_train_loss = []
plotter_test_loss = []

plotter_train_acc = []
plotter_test_acc = []

for epoch in range(1, epochs + 1):
    
    train_loss, train_acc = train_nn.train(trainloader = inputs_train_data, model=model, device=device, optimizer = optimizer,
                                  criterion=criterion, verbose=50)
    val_loss, val_acc = train_nn.validate(valloader= targets_test_data, model=model, device=device, criterion=criterion, verbose=10)

    #test_loss, test_acc =  validate(valloader= test_input_data, model=model, device=device, criterion=criterion, verbose=10)

    #writer.add_scalar(f'Loss/test {pick} {nodes} nodes {network}', val_loss, epoch)

    ##writer.add_scalar(f'Accuracy/test {pick} {nodes} nodes {network}', val_acc, epoch)


    plotter_train_loss.append(train_loss)
    plotter_test_loss.append(val_loss)
    plotter_train_acc.append(train_acc)
    plotter_test_acc.append(val_acc)


    if epoch%1==0 or epoch == 1:
      print(f"epoch {epoch}:, train loss: {train_loss:.4f}, train acc: {train_acc:.4f}, validation loss: {val_loss:.4f}, validation acc: {val_acc:.4f}")


torch.save(model.state_dict(), model_state)

plt.figure(figsize=(5,5))
plt.plot(np.arange(epochs), plotter_train_loss, label = "training loss")
plt.plot(np.arange(epochs), plotter_test_loss, label = "test loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title(f"{pick}")
plt.legend()

plt_train_acc = np.array(plotter_train_acc)*100
plt_test_acc = np.array(plotter_test_acc)*100

plt.figure(figsize=(5,5))
plt.axis([0, epochs, 0, 100])
plt.plot(np.arange(epochs), plt_train_acc, label = "Training Accuracy")
plt.plot(np.arange(epochs), plt_test_acc, label = "Test Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title(f"{pick}")
plt.legend()
plt.show()
