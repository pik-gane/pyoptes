import os
from tkinter import W
from xml.dom import HierarchyRequestErr
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
from pyoptes.optimization.budget_allocation.supervised_learning.utils import Loader as Loader
from pyoptes.optimization.budget_allocation.supervised_learning.utils import processing as process
from pyoptes.optimization.budget_allocation.supervised_learning.utils import model_selection as model_selection
from pyoptes.optimization.budget_allocation.supervised_learning.utils import device as get_device
from pyoptes.optimization.budget_allocation.supervised_learning.utils import training_process as train_nn
import torch
from pyoptes.optimization.budget_allocation import target_function as f

print("Preparing the target function for a random but fixed transmissions network")
# generate a Waxman graph:
waxman = nx.waxman_graph(120)
pos = dict(waxman.nodes.data('pos'))
# convert into a directed graph:
static_network = nx.DiGraph(nx.to_numpy_array(waxman))

# at the beginning, call prepare() once:
f.prepare(
  use_real_data=False, 
  static_network=static_network,
  n_nodes=120,
  max_t=365, 
  expected_time_of_first_infection=30, 
  capacity_distribution = np.random.lognormal, #lambda size: np.ones(size), # any function accepting a 'size=' parameter
  delta_t_symptoms=60
  )

n_trials = 100

n_inputs = f.get_n_inputs()
total_budget = n_inputs

evaluation_parms = { 
        'n_simulations': 1000, 
        'statistic': lambda a: (np.mean(a**2), np.std(a**2)/np.sqrt(a.size)) #lambda a: np.percentile(a, 95)
        }

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hidden_dims = (128, 64, 32, 16)
nodes = 120
pick = "RNN"

model = model_selection.set_model(pick, dim = nodes, hidden_dims = hidden_dims)
model.to(device)

model.load_state_dict(torch.load("/Users/admin/pyoptes/src/waxman_120_sci.pth"))
#for param_tensor in model.state_dict():
#    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

criterion = nn.L1Loss() #mean absolut error

train_input_data = "/Users/admin/pyoptes/src/inputs_waxman_120_sent_sci.csv"
train_targets_data = "/Users/admin/pyoptes/src/targets_waxman_120_sent_sci.csv"
train_data, test_data = process.postprocessing(train_input_data, train_targets_data, split = 5000, grads = False)
inputs_train_data = DataLoader(train_data, batch_size = 128, shuffle=True)
targets_test_data = DataLoader(test_data, batch_size = 128, shuffle=True)

val_loss, val_acc = train_nn.validate(valloader= targets_test_data, model=model, device=device, criterion=criterion, verbose=10)
print(f'\n\nloss of model: {val_loss}, accuray of model: {val_acc}\n\n')

degree_values = sorted(waxman.degree, key=lambda x: x[1], reverse=True)
hd = []
for i in range(10):
  hd.append(degree_values[i][0])

print(f'nodes with highest degree: {hd}\n')

sentinels = hd
#sentinels = [33, 36, 63, 66]
weights = np.zeros(n_inputs)
weights[sentinels] = 1
shares = weights / weights.sum()

x4 = shares * total_budget

model.requires_grad_(False)

test_x, test_y = process.postprocessing(train_input_data, train_targets_data, split = 1000, grads = True)

test_x = test_x.to_numpy()
test_y = test_y.to_numpy()
initial_budget = test_x[10] #?makes a difference wether I use a.e. Sentinel based BudDist as Init or a a.e. random BD

f_eval, si_out_sq_err = f.evaluate(x4, **evaluation_parms)
si_out,  si_out_err = np.sqrt(f_eval), si_out_sq_err/(2*np.sqrt(f_eval)) #

print(f'initial budget (baseline): \n{x4} ...\n')
print(f'baseline top 10 highest degree nodes: {si_out}\n')
print(f'std error baseline top 10 highest degree nodes: {si_out_err}\n')

test_x = torch.tensor(initial_budget).requires_grad_(True)
test_y = torch.tensor(np.zeros_like(test_y[0]))

optimiser = optim.AdamW([test_x], lr= 0.01)

epochs = 1000000
opt_input = []
si_out_0 = np.inf
nn_out_0 = np.inf

for epoch in range(1, epochs + 1):
    val_loss, grads = train_nn.optimise(input_budget = test_x, targets = test_y, 
        model=model, optimiser = optimiser, device=device, criterion=criterion, verbose=10)

    #print(f"epoch: {epoch}, loss: {val_loss}") # accuracy: {val_acc}")
    nn_out = train_nn.evaluate(grads, model = model, device = device)

    f_eval, si_out_sq_err = f.evaluate(x4, **evaluation_parms)
    si_out,  si_out_err = np.sqrt(f_eval), si_out_sq_err/(2*np.sqrt(f_eval)) #

    if si_out < si_out_0:
        print(f"epoch: {epoch}, predicted no. of infected animals for optimised budget NN: {nn_out}")
        print(f"epoch: {epoch}, predicted no. of infected animals for optimised budget SI: {si_out}")
        print(f"epoch: {epoch}, std err predicted no. of infected animals for optimised budget SI: {si_out_err}")
        print("\n")
        si_out_0 = si_out
        opt_input = grads
        pd.DataFrame(grads).T.to_csv("/Users/admin/pyoptes/src/optimal_budget_120_waxman_sent_sci.csv", header = True)
    if epoch%1000==0:
        print(f'\nreached {epoch} epochs')
#print(f'dloss/dx:\n {grads[0][0][0].shape}')
print(opt_input)