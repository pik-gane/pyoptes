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
from ray import tune
from scipy.stats import gaussian_kde as kde

print(f.__doc__)
# set some seed to get reproducible results:
set_seed(1)

# generate a Waxman graph:
waxman = nx.waxman_graph(120)
pos = dict(waxman.nodes.data('pos'))
# convert into a directed graph:
static_network = nx.DiGraph(nx.to_numpy_array(waxman))

evaluation_parms = { 
        'n_simulations': 1000, 
        'statistic': lambda a: np.mean(a**2)
        }

# at the beginning, call prepare() once:
f.prepare(
  use_real_data=False, 
  static_network=None,
  n_nodes=120,
  max_t=365, 
  expected_time_of_first_infection=30, 
  capacity_distribution = np.random.lognormal, #lambda size: np.ones(size), # any function accepting a 'size=' parameter
  delta_t_symptoms=60
  )

n_inputs = f.get_n_inputs()
print("n_inputs (=number of network nodes):", n_inputs)

total_budget = 1.0 * n_inputs  # i.e., on average, nodes will do one test per year
n_trials = 100 #run 100x f.evaluate(runs 1000x) to get mean


"""baseline - n/12 sentinels"""
no_sent = np.int(n_inputs/12)
sentinels = list(np.random.choice(np.arange(0, n_inputs), no_sent, replace=False))
weights = np.zeros(n_inputs)
weights[sentinels] = 1
shares = weights / weights.sum()

"""on barabasi network"""
x_bara = shares * total_budget
ys_bara =  np.array([f.evaluate(x_bara, **evaluation_parms) for it in range(n_trials)])
#ys_bara = np.mean(ys_bara)
log_bara = np.log(ys_bara)

"""baseline - n/12 sentinels on a waxman network"""
f.prepare(
  use_real_data=False, 
  static_network=static_network,
  n_nodes=120,
  max_t=365, 
  expected_time_of_first_infection=30, 
  capacity_distribution = np.random.lognormal, #lambda size: np.ones(size), # any function accepting a 'size=' parameter
  delta_t_symptoms=60
  )

x_waxman = shares * total_budget
ys_waxman =  np.array([f.evaluate(x_waxman, **evaluation_parms) for it in range(n_trials)])
#ys_waxman = np.mean(ys_waxman)
log_waxman = np.log(ys_waxman)

print(np.sqrt(np.mean(ys_waxman)), np.sqrt(np.mean(ys_bara)))

xs = np.linspace(ys_waxman.min(), ys_waxman.max())
plt.figure()
plt.plot(xs, kde(ys_waxman)(xs), label="waxman n/12 sentinels")
plt.legend()
plt.title("distribution of f(x) for different fixed inputs x waxman")

xs = np.linspace(ys_bara.min(), ys_bara.max())
plt.figure()
plt.plot(xs, kde(ys_bara)(xs), alpha=0.5, label="bara n/12 sentinels")
plt.legend()
plt.title("distribution of f(x) for different fixed inputs x barabasi")


xs = np.linspace(log_waxman.min(), log_waxman.max())
plt.figure()
plt.plot(xs, kde(log_waxman)(xs), alpha=0.5, label="log of waxman n/12 sentinels")
plt.legend()
plt.title(" distribution of log f(x) for different fixed inputs x waxman")

xs = np.linspace(log_bara.min(), log_bara.max())
plt.figure()
plt.plot(xs, kde(log_bara)(xs), label="log of barabasi n/12 sentinels")
plt.legend()
plt.title(" distribution of log f(x) for different fixed inputs x bara")


plt.show()