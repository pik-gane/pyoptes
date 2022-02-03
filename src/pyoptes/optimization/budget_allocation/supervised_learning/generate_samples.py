"""
Simple test to illustrate how the target function could be used.
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

warnings.filterwarnings("ignore")

print(f.__doc__)

# set some seed to get reproducible results:
set_seed(1)

print("Preparing the target function for a lattice-based, fixed transmissions network")

# generate a 11 by 11 2d lattice with nodes numbered 0 to 120:
lattice = nx.DiGraph(nx.to_numpy_array(nx.lattice.grid_2d_graph(11, 11)))

# at the beginning, call prepare() once:
f.prepare(
  use_real_data=False, 
  static_network=lattice,
  n_nodes=121,
  max_t=365, 
  expected_time_of_first_infection=30, 
  capacity_distribution=lambda size: np.ones(size), # any function accepting a 'size=' parameter
  delta_t_symptoms=60
  )

n_inputs = f.get_n_inputs()
print("n_inputs (=number of network nodes):", n_inputs)

total_budget = 1.0 * n_inputs

samples = 1000
n_trials = 1000

#plt.figure()
#nx.draw_kamada_kawai(lattice, node_color=[[0,0,0,xi/xmax] for xi in x])
#plt.show()

evaluation_parms = { 
        'n_simulations': 100, 
        'statistic': lambda a: np.percentile(a, 95)
        }

for i in range(samples):
  
  # x that is randomly sampled
  weights = np.random.rand(n_inputs)
  shares = weights / weights.sum()
  x_rnd = np.round((shares * total_budget), 2)  
  #print(x_rnd)
  ys_rnd = np.array([f.evaluate(x_rnd, **evaluation_parms) for it in range(n_trials)])
  ys_rnd = np.round(np.mean(ys_rnd), 2)
  
  # x that is based on total capacity of node
  #weights = f.capacities
  #shares = weights / weights.sum()
  #x_cap = shares * total_budget
  #ys_cap = np.array([f.evaluate(x_cap, **evaluation_parms) for it in range(n_trials)])
  #ys_cap = np.mean(ys_cap)

  # do the same for an x that is based on the total number of incoming transmissions per node:
  #target_list = f.model.transmissions_array[:, 3]
  #values, counts = np.unique(target_list, return_counts=True)
  #weights = np.zeros(n_inputs)
  #weights[values] = counts
  #shares = weights / weights.sum()
  #total_budget = 1.0 * n_inputs
  #x_trans = shares * total_budget
  #ys_trans = np.array([f.evaluate(x_trans, **evaluation_parms) for it in range(n_trials)])
  #ys_trans = np.mean(ys_trans)

  # x that is based on sentinels
  #x = np.zeros(n_inputs)
  #x[-1] = total_budget  
  
  no_sent = np.random.choice(np.arange(0, n_inputs), 1) #get a value between 0 and n_inputs

  sentinels = list(np.random.choice(np.arange(0,n_inputs), no_sent, replace=False)) #get no_sents number of sentinels between 0 and n_inputs
  
  weights = np.zeros(n_inputs)
  weights[sentinels] = 1
  shares = weights / weights.sum()

  x_sent = np.round((shares * total_budget), 2)
  #print(x_sent)
  
  ys_sent =  np.array([f.evaluate(x_sent, **evaluation_parms) for it in range(n_trials)])
  ys_sent = np.round(np.mean(ys_sent), 2)

  # do the same for an x that is based on the static network's node degrees:
  #weights = np.array(list([d for n, d in f.network.degree()]))
  #shares = weights / weights.sum()
  #x_degree = shares * total_budget
  #ys_degree = np.array([f.evaluate(x_degree, **evaluation_parms) for it in range(n_trials)])
  #ys_degree = np.mean(ys_degree)
  
  f_1 = open ('/content/drive/MyDrive/pyoptes/src/input_data.csv', 'a')
  writer = csv.writer(f_1)
  writer.writerow(x_rnd)
  #writer.writerow(x_cap)
  #writer.writerow(x_trans)
  writer.writerow(x_sent)
  #writer.writerow(x_degree)
  f_1.close()

  f_2 = open ('/content/drive/MyDrive/pyoptes/src/label_data.csv', 'a')
  f_2.write(np.str(ys_rnd) + "\n" +  np.str(ys_sent) + "\n")
  #np.str(ys_cap) + "\n" + np.str(ys_trans) + "\n" + np.str(ys_degree)
  f_2.close()

  print(f'\ngenerated: {(i+1)*2} pairs of datapoints x and y')

"""
for i in tqdm(range(n_training_points)):
    
    #weights = np.random.rand(n_inputs)
    #shares = weights / weights.sum()
    x = np.zeros(n_inputs)
    x[-1] = total_budget  
    print(x)
    #sentinels = list(np.random.choice(np.arange(0,121), 16, replace=False))
    #sentinels = [0, 3, 6, 9, 30, 33, 36, 39, 60, 63, 66, 69, 90, 93, 96, 99]
    #sentinels = [33, 36, 63, 66]
    #weights = np.zeros(121)
    #weights[sentinels] = 1
    #shares = weights / weights.sum()

    x = shares * total_budget

    print(f'\nOne evaluation: {f.evaluate(x, **evaluation_parms)}')  # to focus on the tail of the distribution)

    y =  np.array([f.evaluate(x, **evaluation_parms) for it in range(n_trials)])

    y = np.mean(y)
    
    print(f'\nAverage over 1000 evaluations: {y}')  # to focus on the tail of the distribution)

    
    f_1 = open ('/content/drive/MyDrive/pyoptes/src/input_data_rnd_2.csv', 'a')
    writer = csv.writer(f_1)
    writer.writerow(x)
    f_1.close()

    f_2 = open ('/content/drive/MyDrive/pyoptes/src/label_data_rnd_2.csv', 'a')
    y = np.str(y)
    f_2.write(y+"\n")
    f_2.close()

    print(f'\ngenerated: {i+1} pairs of datapoints x and y')
    """