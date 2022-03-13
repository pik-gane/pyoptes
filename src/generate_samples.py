"""
Simple demonstration to illustrate how to generate training samples
"""

from operator import xor
import numpy as np
from scipy.stats import gaussian_kde as kde
import pylab as plt
from pyoptes import set_seed
from pyoptes.optimization.budget_allocation import target_function as f
import pandas as pd
from tqdm import tqdm
import csv
import warnings
import networkx as nx
from scipy.stats import lognorm

warnings.filterwarnings("ignore")

# set some seed to get reproducible results:
set_seed(1)

# define size of our network
n_nodes = 120

# define distribution of our node test capacities 
def caps(size): 
  return lognorm.rvs(s=2, scale=np.exp(4), size=size)

# generate a 11 by 11 2d lattice with nodes numbered 0 to 120:
lattice = nx.DiGraph(nx.to_numpy_array(nx.lattice.grid_2d_graph(11, 11)))

# generate a Waxman graph:
waxman = nx.waxman_graph(n_nodes)
pos = dict(waxman.nodes.data('pos'))
# convert into a directed graph:
static_network = nx.DiGraph(nx.to_numpy_array(waxman))

# at the beginning, call prepare() once:
f.prepare(
  use_real_data=False, #False = synthetic data
  static_network=static_network, #use waxman graph
  n_nodes=n_nodes, #size of network
  max_t=365, #time horizon
  expected_time_of_first_infection=30, #30 days
  capacity_distribution = caps, #lambda size: np.ones(size), # any function accepting a 'size=' parameter
  delta_t_symptoms=60 #abort simulation after 60 days when symptoms first show up and testing becomes obsolet
  )

n_inputs = f.get_n_inputs() #number of nodes of our network

print("n_inputs (=number of network nodes):", n_inputs)

total_budget = 1.0 * n_inputs #total budget equals number of nodes
n_samples = 1 #how many samples we want to generate
n_simulations = 1000 #run n_simulations of our target function to reduce std error
num_cpu_cores = -1 #use all cpu cores

#evaluation params of our target function
evaluation_parms = { 
        'n_simulations': n_simulations, 
        'parallel': True,
        'num_cpu_cores': num_cpu_cores
        }
    
samples_per_iteration = 3

# generate n_samples       
for i in range(n_samples):

  #x that is based on sentinels
  no_sent = np.random.choice(np.arange(0, n_inputs), 1) #choose a number no_sent between 1 and number of nodes
  sentinels = list(np.random.choice(np.arange(0,n_inputs), no_sent, replace=False)) #decide which nodes are sentinels
  weights = np.zeros(n_inputs) 
  weights[sentinels] = 1 
  shares = weights / weights.sum()
  x_sent = shares * total_budget #distribute budget among sentinels
  (dmg_node_sent, se_node_sent) = f.evaluate(x_sent, **evaluation_parms)
  #print(np.sum(se_node_sent)/n_nodes)

  #x that is randomly sampled
  weights = np.random.rand(n_inputs)
  shares = weights / weights.sum()
  x_rnd = shares * total_budget  
  (dmg_node_rnd, se_node_rnd) = f.evaluate(x_rnd, **evaluation_parms)
  #print(np.sum(se_node_rnd)/n_nodes)

  # x that is exp. randomly sampled
  weights = np.random.exponential(size = n_inputs)
  shares = weights / weights.sum()
  x_exp = shares * total_budget  
  (dmg_node_exp, se_node_exp) = f.evaluate(x_exp, **evaluation_parms)
  #print(np.sum(se_node_exp)/n_nodes)


  #write our inputs to csv
  f_1 = open ('/Users/admin/pyoptes/src/pyoptes/optimization/budget_allocation/supervised_learning/data_per_node/wx_inputs.csv', 'a')
  writer = csv.writer(f_1)
  writer.writerow(x_rnd)
  writer.writerow(x_exp)
  writer.writerow(x_sent)
  f_1.close()

  #write our targets to csv
  f_2 = open ('/Users/admin/pyoptes/src/pyoptes/optimization/budget_allocation/supervised_learning/data_per_node/wx_targets.csv', 'a')
  writer = csv.writer(f_2)
  writer.writerow(dmg_node_rnd)
  writer.writerow(dmg_node_exp)
  writer.writerow(dmg_node_sent)
  #f_2.write(np.str(dmg_node_rnd) + "\n" + np.str(dmg_node_exp) + "\n" + np.str(dmg_node_sent) + "\n")
  f_2.close()

  #write our standard errors to csv (optional)
  f_3 = open ('/Users/admin/pyoptes/src/pyoptes/optimization/budget_allocation/supervised_learning/data_per_node/wx_standard_errors.csv', 'a')
  writer = csv.writer(f_3)
  writer.writerow(se_node_rnd)
  writer.writerow(se_node_exp)
  writer.writerow(se_node_sent)
  #f_3.write(np.str(se_node_rnd) + "\n" + np.str(se_node_exp) + "\n" + np.str(se_node_sent) + "\n")
  f_3.close()

  print(f'\ngenerated: {(i+1)*samples_per_iteration} pairs of datapoints x and y')