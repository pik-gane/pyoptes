import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from sklearn.metrics import explained_variance_score, mean_squared_error
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
from pyoptes.optimization.budget_allocation.supervised_learning import NN as nets
from pyoptes.optimization.budget_allocation.supervised_learning.utils import Loader as Loader

def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device
device = get_device()

def postprocessing(train_input_data, train_targets_data, split):
    
    train_input_data = pd.read_csv(train_input_data, header = None, sep = ',')
    train_targets_data = pd.read_csv(train_targets_data, header = None, sep = ',')
    
    is_NaN = train_input_data.isnull()
    row_has_NaN = is_NaN.any(axis=1)
    rows_with_NaN = train_input_data[row_has_NaN]
    del_cells = rows_with_NaN.index.values
    train_input_data = train_input_data.drop(del_cells)
    train_targets_data = train_targets_data.drop(del_cells)

    subset_training_input = train_input_data.iloc[split:]
    subset_training_targets = train_targets_data.iloc[split:]

    subset_val_input = train_input_data.iloc[0:split]
    subset_val_targets = train_targets_data.iloc[0:split]

    return subset_training_input, subset_training_targets, subset_val_input, subset_val_targets

train_input_data = "/Users/admin/pyoptes/src/pyoptes/optimization/budget_allocation/supervised_learning/input_data_none.csv"
train_targets_data = "/Users/admin/pyoptes/src/pyoptes/optimization/budget_allocation/supervised_learning/targets_data_none.csv"

test_input_data = "/Users/admin/pyoptes/src/pyoptes/optimization/budget_allocation/supervised_learning/input_data_round.csv"
test_targets_data = "/Users/admin/pyoptes/src/pyoptes/optimization/budget_allocation/supervised_learning/targets_data_round.csv"

train_input, train_targets, val_input, val_targets = postprocessing(train_input_data, train_targets_data, split = 300)

print(f'Size of training inputs, targets: {len(train_input)} \nSize of test inputs, targets: {len(val_input)}')

training_set = Loader(input_path = train_input, 
                      targets_path = train_targets, path = False)

validation_set = Loader(input_path = val_input, 
                        targets_path = val_targets, path = False)

train_data = DataLoader(training_set, batch_size = 32, shuffle=True)
test_input_data = DataLoader(validation_set, batch_size = 32, shuffle=True)


def train(trainloader: DataLoader, model: torchvision.models, device: torch.device, 
          optimizer: torch.optim, criterion: torch.nn , verbose: int = 20):

    model.train()
    true = []
    pred = []
    train_loss = []

    for i, (inputs, targets) in enumerate(trainloader, 1):
        

        inputs, targets = inputs.to(device).float(), targets.to(device).float()

        optimizer.zero_grad()

        output = model.forward(inputs)
        loss = criterion(output, targets)

        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())

        for j, val in enumerate(output):
            true.append(targets[j].item())
            pred.append(output[j].item())

    acc = explained_variance_score(true, pred)

    return np.mean(train_loss), acc


def validate(valloader: DataLoader, model: torchvision.models,device: torch.device, 
             criterion: torch.nn , verbose: int = 20):
    
    model.eval()
    true = []
    pred = []
    val_loss = []
    acc = []

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(valloader, 1):

            inputs, targets = inputs.to(device).float(), targets.to(device).float()
            output = model.forward(inputs)
            loss = criterion(output, targets)
            val_loss.append(loss.item())

            for j, val in enumerate(output):
                true.append(targets[j].item())
                pred.append(output[j].item())
    
    acc = explained_variance_score(true, pred) #1 - var(y-y_hat)/var(y) 

    return np.mean(val_loss), acc
    
random.seed(10)

epochs = 150

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_model(model = str, dim = int):
    if model == "Linear":
        model = nets.LinearNetwork(dim, 1, bias = True)
    elif model == "RNN":
        model = nets.RNNetwork(dim, 1, bias = True)
    elif model == "FCN":
        model = nets.FCNetwork(dim, 1, bias = True)
    #elif model == "GRU":
    #    model = nets.GRU(dim, 1, bias = True)
    return model

pick = "RNN"

model = set_model(pick, dim = 121)

model.to(device)

#criterion = nn.MSELoss() 
criterion = nn.L1Loss() #mean absolut error

#learning_rate = 3.5

"""opt_params for Adam/W"""
optimizer_params = {"lr": 0.01, "weight_decay": 0.005, "betas": (0.9, 0.999)}

"""opt_params for SGD"""
#optimizer_params = {"lr": 0.02, "weight_decay": 0.0005, "momentum": True}

"""opt_params for Adagrad"""
#optimizer_params = {"lr": 0.1}

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

    train_loss, train_acc = train(trainloader = train_data, model=model, device=device, optimizer = optimizer,
                                  criterion=criterion, verbose=50)
    
    val_loss, val_acc = validate(valloader= test_input_data, model=model, device=device, criterion=criterion, verbose=10)


    #test_loss, test_acc =  validate(valloader= test_input_data, model=model, device=device, criterion=criterion, verbose=10)


    plotter_train_loss.append(train_loss)
    plotter_test_loss.append(val_loss)


    plotter_train_acc.append(train_acc)
    plotter_test_acc.append(val_acc)


    if epoch%1==0 or epoch == 1:
      print(f"epoch {epoch}:, train loss: {train_loss:.4f}, train acc: {train_acc:.4f}, validation loss: {val_loss:.4f}, validation acc: {val_acc:.4f}")

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
