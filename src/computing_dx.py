from email.mime import base
import os
from tkinter import W
from xml.dom import HierarchyRequestErr
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import pylab as plt
from pyoptes import set_seed
from pyoptes.optimization.budget_allocation import target_function as f
import networkx as nx
from pyoptes.optimization.budget_allocation.supervised_learning.utils import Loader as Loader
from pyoptes.optimization.budget_allocation.supervised_learning.utils import processing as process
from pyoptes.optimization.budget_allocation.supervised_learning.utils import model_selection as model_selection
from pyoptes.optimization.budget_allocation.supervised_learning.utils import training_process as train_nn
from pyoptes.optimization.budget_allocation.supervised_learning.utils import distribution as dcaps
import torch
from pyoptes.optimization.budget_allocation import target_function as f
from scipy.stats import lognorm
from pyoptes import set_seed

def caps(size): 
  return lognorm.rvs(s=2, scale=np.exp(4), size=size)

set_seed(1)
n_nodes = 120

print("Preparing the target function for a random but fixed transmissions network")
# generate a Waxman graph:
waxman = nx.waxman_graph(n_nodes)
pos = dict(waxman.nodes.data('pos'))
# convert into a directed graph:
static_network = nx.DiGraph(nx.to_numpy_array(waxman))

# at the beginning, call prepare() once:
f.prepare(
  use_real_data=False, #False = synthetic data
  static_network= None, #static_network, #use waxman graph
  n_nodes= n_nodes, #size of network
  max_t=365, #time horizon
  expected_time_of_first_infection=30, #30 days
  capacity_distribution = caps, #lambda size: np.ones(size), # any function accepting a 'size=' parameter
  delta_t_symptoms=60 #abort simulation after 60 days when symptoms first show up and testing becomes obsolet
  )

n_inputs = f.get_n_inputs() #number of nodes of our network

print("n_inputs (=number of network nodes):", n_inputs)

total_budget = 1.0 * n_inputs #total budget equals number of nodes
n_simulations = 100000 #run n_simulations of our target function to reduce std error
num_cpu_cores = -1 #use all cpu cores

#evaluation params of our target function
evaluation_parms = { 
        'n_simulations': n_simulations, 
        'parallel': True,
        'num_cpu_cores': num_cpu_cores
        }

"""define NN model parameters, load pre-trained model state"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_dim = 120
layer_dimensions = (128, 64, 32, 16, output_dim)
pick = "FCN"
model = model_selection.set_model(pick, dim = n_nodes, layer_dimensions = layer_dimensions)
model.to(device)
model.requires_grad_(False) #freeze weights
#model.load_state_dict(torch.load(f"/Users/admin/pyoptes/src/pyoptes/optimization/budget_allocation/supervised_learning/trained_nn_states/wx_120_{pick}_per_node.pth"))
model.load_state_dict(torch.load(f"/Users/admin/pyoptes/src/pyoptes/optimization/budget_allocation/supervised_learning/trained_nn_states/ba_120_{pick}_per_node.pth"))

criterion = nn.L1Loss() #mean absolut error

"""evaluate accuracy pre-trained model"""
#inputs = "/Users/admin/pyoptes/src/pyoptes/optimization/budget_allocation/supervised_learning/training_data/wx_inputs.csv"
#targets = "/Users/admin/pyoptes/src/pyoptes/optimization/budget_allocation/supervised_learning/training_data/wx_targets.csv"

inputs = "//Users/admin/pyoptes/src/pyoptes/optimization/budget_allocation/supervised_learning/training_data/ba_inputs_100k.csv"
targets = "/Users/admin/pyoptes/src/pyoptes/optimization/budget_allocation/supervised_learning/training_data/ba_targets_100k.csv"

test_x, test_y = process.postprocessing(inputs, targets, split = 1000, grads = True)
test_x = test_x.to_numpy()
test_y = test_y.to_numpy()
train_data, test_data = process.postprocessing(inputs, targets, split = 1000, grads = False)
inputs_train_data = DataLoader(train_data, batch_size = 128, shuffle=True)
targets_test_data = DataLoader(test_data, batch_size = 128, shuffle=True)
val_loss, val_acc = train_nn.validate(valloader= targets_test_data, model=model, device=device, criterion=criterion, verbose=10)
print(f'\n\nloss of model: {val_loss}, accuray of model: {val_acc}\n\n')


"""baseline 10 sentinels highest degree"""
degree_values = sorted(static_network.degree, key=lambda x: x[1], reverse=True)
hd = []
for i in range(10):
  hd.append(degree_values[i][0])
print(f'nodes with highest degree: {hd}\n')
sentinels = hd
weights = np.zeros(n_inputs)
weights[sentinels] = 1
shares = weights / weights.sum()
x4 = shares * total_budget

"""initial budget"""
initial_budget = test_x[1]
print(f'initial budget: {initial_budget[:6]}...') #?makes a difference wether I use a.e. Sentinel based BudDist as Init or a a.e. random BD
(dmg_node, se_node) = f.evaluate(initial_budget, **evaluation_parms)
si_out_initial = np.sum(dmg_node)
si_se_initial = np.sum(se_node)
si_inf_initial = np.sqrt(si_out_initial)
si_se_total = si_se_initial/(2*si_inf_initial )
initial_rel_se = si_se_total/si_inf_initial 

"""evaluate budget test once p.a"""
test_1_per_year = np.ones_like(initial_budget)
(dmg_node, se_node) = f.evaluate(test_1_per_year, **evaluation_parms)
si_out_y = np.sum(dmg_node)
si_se_y = np.sum(se_node)
si_inf_year = np.sqrt(si_out_y)
si_se_total = si_se_y/(2*si_inf_year)
year_rel_se = si_se_total/si_inf_year

"""evaluate baseline budget"""
(dmg_node, se_node) = f.evaluate(x4, **evaluation_parms)
si_out_base = np.sum(dmg_node)
si_se_base = np.sum(se_node)
si_base_total = np.sqrt(si_out_base)
si_se_total = si_se_base/(2*si_base_total)
base_rel_se = si_se_total/si_base_total

print(f'\nbaseline initial budget: {si_inf_initial}, std error: {initial_rel_se}\n')
print(f'baseline test once per year: {si_inf_year}, std error: {year_rel_se}\n')
print(f'baseline top 10 highest degree nodes: {si_base_total}, std error: {base_rel_se}\n')

initial_budget = torch.tensor(initial_budget).requires_grad_(True) #we allow our inputs to have gradients
target_is_zero = torch.tensor(np.zeros_like(test_y[0]))

"""hyper-parameters"""
lr = -2.5
optimiser = optim.AdamW([initial_budget], lr = 10**(lr))
epochs = 100000

opt_input = []
plot_nn = []
plot_si = []
imp_animals = []
plot_se = []

baseline_animals = si_base_total
nn_base = baseline_animals

for epoch in range(1, epochs + 1):

    val_loss, grads = train_nn.optimise(input_budget = initial_budget, targets = target_is_zero, 
        model=model, optimiser = optimiser, device=device, criterion=criterion, verbose=10)  
    #print(f"epoch: {epoch}, loss: {val_loss}") # accuracy: {val_acc}")
    nn_out = train_nn.evaluate(grads, model = model, device = device)
    nn_out = nn_out.squeeze()
    nn_inf_animals = np.sqrt(torch.sum(nn_out).item())
    initial_budget = grads.clone().detach().requires_grad_(True)
    #print(epoch, nn_inf_animals)
    if epoch%10000==0:  
      print(epoch)
      (dmg_node_per_node, se_node) = f.evaluate(grads, **evaluation_parms)
      si_out_total_dmg = np.sum(dmg_node_per_node)
      si_inf_animals = np.sqrt(si_out_total_dmg)
      si_out_total_se = np.sum(se_node)
      si_se_total = si_out_total_se/(2*np.sqrt(si_out_total_dmg))
      rel_se = si_se_total/si_inf_animals
      
      print(f'nn: {nn_inf_animals}, si: {si_inf_animals}')

    plot_nn.append(nn_inf_animals)
    #plot_si.append(si_inf_animals)
    #plot_se.append(si_se_total)
    #test_x = torch.tensor(grads).requires_grad_(True)
    """
    if si_inf_animals < baseline_animals:
      #(dmg_node_per_node, se_node) = f.evaluate(grads, **evaluation_parms)
      #si_out_total_dmg = np.sum(dmg_node_per_node)
      #si_inf_animals = np.sqrt(si_out_total_dmg)
      #si_out_total_se = np.sum(se_node)
      #si_se_total = si_out_total_se/(2*np.sqrt(si_out_base))
      #rel_se = si_se_total/si_inf_animals
      #print(f'NN prediction: {nn_inf_animals}')
      #si_out_0 = si_out_dmg
      #opt_input = grads
      pd.DataFrame(grads).T.to_csv("/Users/admin/pyoptes/src/optimal_budget_120_waxman_sent_sci.csv", header = True)
      #imp_animals.append(si_out_dmg-si_out_0)) 
      print(f"epoch: {epoch}, predicted no. of infected animals for optimised budget NN: {nn_inf_animals}")
      print(f"epoch: {epoch}, predicted no. of infected animals for optimised budget SI: {si_inf_animals}")
      print(f"epoch: {epoch}, std err predicted no. of infected animals for optimised budget SI: {rel_se}")
      
      improv_by = (1-si_inf_animals/si_base_total)

      if 1-np.sqrt(si_inf_animals)/baseline_animals < 0:
        print(f'worse by {-(1-si_inf_animals/si_base_total)}% compared to baseline')
        print("\n")
      elif 1-np.sqrt(si_inf_animals)/baseline_animals > 0:
        print(f'improved by {improv_by}% compared to baseline')
        print("\n")

      #to_write = np.concatenate((grads, si_inf_animals, rel_se, improv_by), axis = None)
      #f_1 = open('/Users/admin/pyoptes/src/optimal_budget.csv', 'a')
      #writer = csv.writer(f_1)
      #writer.writerow(to_write)
      #f_2.write(np.str(dmg_node_rnd) + "\n" + np.str(dmg_node_exp) + "\n" + np.str(dmg_node_sent) + "\n")
      #f_1.close()

      baseline_animals = si_inf_animals
      #nn_base = nn_inf_animals
"""
    """
    if epoch%10==0:
        f_ev = []
        s_i = []

        evaluation_parms = { 
          'n_simulations': 10000, 
          'parallel': True,
          'num_cpu_cores': num_cpu_cores
          }
    
        print(f'\nreached {epoch} epochs\n')
        
        
        (f_eval, si_out_sq_err) = f.evaluate(grads, **evaluation_parms)
        print(np.sqrt(np.sum(f_eval)), np.sqrt(np.sum(si_out_sq_err)))
      """  

#print(f'dloss/dx:\n {grads[0][0][0].shape}')
#print(opt_input)

plt.figure(figsize=(5,5))
#plt.axis([0, epochs, 0, 100])
plt.plot(np.arange(start= 1, stop = epochs+1), plot_nn, label = "NN Predictions")
#plt.plot(np.arange(start= 1, stop = epochs+1), plot_si, label = "SI Predictions")
#plt.plot(np.arange(start= 1, stop = epochs+1), plot_se, label = "Standard error")
plt.xlabel("Epochs")
plt.ylabel("number of infected animals")
plt.title(f"Budget allocation optimisation, wx, 120 nodes, lr: {lr}")
plt.legend()

#plt.figure(figsize=(5,5))
#plt.axis([0, epochs, 0, 100])
#plt.plot(np.arange(epochs), imp_animals, label = "Improve animals")
#plt.xlabel("Epochs")
#plt.ylabel("number of infected animals")
#plt.title(f"Budget allocation optimisation")
#plt.legend()

plt.show()