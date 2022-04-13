"""
Simple demonstration to illustrate how to generate training samples
"""

from operator import xor
import numpy as np
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

#load pre-generated networks
transmissions_wx = pd.read_csv(f"../data/waxman_networks/{n_nodes}/WX0/transmissions.txt", header = None)
capacities_wx = pd.read_csv(f"../data/waxman_networks/{n_nodes}/WX0/capacity.txt", header = None)
capacities_wx = capacities_wx.to_numpy().squeeze()
transmissions_wx = transmissions_wx.to_numpy()
capacities_wx = np.int_(capacities_wx)

# at the beginning, call prepare() once:
f.prepare(
  use_real_data = False, #False = synthetic data
  static_network = None, #use waxman graph
  n_nodes = n_nodes, #size of network
  max_t = 365, #time horizon
  expected_time_of_first_infection = 30, #30 days
  capacity_distribution = capacities_wx, #lambda size: np.ones(size), # any function accepting a 'size=' parameter
  delta_t_symptoms=60, #abort simulation after 60 days when symptoms first show up and testing becomes obsolet
  p_infection_by_transmission=0.5,
  pre_transmissions = transmissions_wx #use pre-stored transmission data
  )

#parameter for f.evaluation()
n_inputs = f.get_n_inputs() #number of nodes of our network
print("n_inputs (=number of network nodes):", n_inputs)
n_simulations = 100000 #run n_simulations of our target function to reduce std error
num_cpu_cores = -1 #use all cpu cores


def est_prob_and_stderr(is_infected):
    ps = np.mean(is_infected, axis=0)
    stderrs = np.sqrt(ps * (1-ps) / is_infected.shape[0])
    return (ps, stderrs)

def n_infected_animals(is_infected):
    return np.sum(f.capacities * is_infected)

def n_infected_animals_node(is_infected):
    return f.capacities * is_infected


def shares(is_infected):
    is_inf = f.model.detection_was_by_test
    return is_inf

def share_detected(is_inf):
    detected = np.count_nonzero(is_inf == True)
    ratio = detected/np.size(is_inf)
    return ratio

def mean_square_and_stderr(n_infected_animals):
    values = n_infected_animals**2
    estimate = np.mean(values, axis=0)
    stderr = np.std(values, ddof=1, axis=0) / np.sqrt(values.shape[0])
    return estimate, stderr

def mean_and_stderr(n_infected_animals):
    values = n_infected_animals
    estimate = np.mean(values, axis=0)
    stderr = np.std(values, ddof=1, axis = 0) / np.sqrt(np.size(values))
    return estimate, stderr

def p_95(n_infected_animals):
    values = n_infected_animals
    estimate = np.percentile(values, 95, axis = 0)
    stderr = np.std(values, ddof=1, axis=0)
    return estimate, stderr

    
#evaluation params of our target function

#generate samples: 
  #mean_square_and_stderr = np.mean(a**2)
  #np.mean
  #lambda a: np.percentile(a, 95)
  #share detected 

node_evaluation_params = { 
        'aggregation' : n_infected_animals_node,
        'statistic' : est_prob_and_stderr, #focus on the 2nd moment of the distribution: statistic=lambda a: np.mean(a**2)
        'n_simulations' : n_simulations, 
        'parallel': True,
        'num_cpu_cores': num_cpu_cores
        }


mean_sq_evaluation_params = { 
        'aggregation' : n_infected_animals,
        'statistic' : mean_square_and_stderr, #focus on the 2nd moment of the distribution: statistic=lambda a: np.mean(a**2)
        'n_simulations' : n_simulations, 
        'parallel': True,
        'num_cpu_cores': num_cpu_cores
        }

p95_evaluation_params = { 
        #'aggregation' : n_infected_animals,
        'statistic' : p_95, #focus on tail of distribution
        'n_simulations' : n_simulations, 
        'parallel': True,
        'num_cpu_cores': num_cpu_cores
        }

mean_evaluation_params = { 
        'aggregation' : n_infected_animals,
        'statistic' : mean_and_stderr, #computes the mean
        'n_simulations' : n_simulations, 
        'parallel': True,
        'num_cpu_cores': num_cpu_cores
        }

share_detected_evaluation_params = {
        'aggregation': shares,
        'statistic' : share_detected,
         'n_simulations' : n_simulations, 
        'parallel': True,
        'num_cpu_cores': num_cpu_cores
        }

budget_distributions = pd.read_csv("pyoptes/optimization/budget_allocation/supervised_learning/budgets/budget_120_4.0N.csv")
print(f'Size of dataset: {len(budget_distributions)}')


# generate n_samples       
for i in tqdm(range(len(budget_distributions))):

  budget_sample = budget_distributions.iloc[i]

  shares_det = f.evaluate(budget_sample, **share_detected_evaluation_params)

  prob_node, se_node = f.evaluate(budget_sample, **node_evaluation_params) #prob infection per node
  stderr_node = np.sum(se_node)/np.size(prob_node)

  (total_dmg_mean_sq, total_se_mean_sq) = f.evaluate(budget_sample, **mean_sq_evaluation_params) #mean_evaluation_params
  stderr_mean_sq = total_se_mean_sq/total_dmg_mean_sq
    
  (total_dmg_mean, total_se_mean) = f.evaluate(budget_sample, **mean_evaluation_params)
  stderr_mean = total_se_mean/total_dmg_mean

  #print(total_dmg_mean, total_se_mean/total_dmg_mean)
  
  targets = np.concatenate((prob_node, total_dmg_mean, total_dmg_mean_sq, shares_det), axis = None)
  targets_se = np.concatenate((se_node, stderr_mean_sq, stderr_mean), axis = None)

  #write our targets to csv
  f_1 = open (f'pyoptes/optimization/budget_allocation/supervised_learning/targets/{np.size(budget_sample)}_1N.csv', 'a')
  writer = csv.writer(f_1)
  writer.writerow(targets)
  f_1.close()

  #write our standard errors to csv
  f_2 = open (f'pyoptes/optimization/budget_allocation/supervised_learning/targets/{np.size(budget_sample)}_1N_se.csv', 'a')
  writer = csv.writer(f_2)
  writer.writerow(targets_se)
  f_2.close()