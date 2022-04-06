from operator import index, xor
from matplotlib.pyplot import axis
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
# from gnn import get_features
import os
from tqdm import tqdm

set_seed(1)


# define distribution of our node test capacities 
def caps(size): 
  return lognorm.rvs(s=2, scale=np.exp(4), size=size)


#args waxman, barabasi
def create_graph(graph: str, number: int, n_nodes: int):

  if graph== "waxman":

    #create a waxman graph, preapre target function for graph, store transmission network waxman
    for n in tqdm(range(number)):
        
        waxman = nx.waxman_graph(n_nodes)
        print(waxman)
        #pos = dict(waxman.nodes.data('pos'))
        # convert into a directed graph:
        static_network = nx.DiGraph(nx.to_numpy_array(waxman))
        print(static_network)

      # at the beginning, call prepare() once:
        f.prepare(
        use_real_data = False, #False = synthetic data
        static_network = static_network, #use waxman graph
        n_nodes = n_nodes, #size of network
        max_t = 365, #time horizon
        expected_time_of_first_infection = 30, #30 days
        capacity_distribution = caps, #lambda size: np.ones(size), # any function accepting a 'size=' parameter
        delta_t_symptoms=60, #abort simulation after 60 days when symptoms first show up and testing becomes obsolet
        p_infection_by_transmission=0.5,
        pre_transmissions= None
        )
        #print(static_network)
        capacities = pd.DataFrame(f.capacities)
        capacities = capacities.T

        degrees = pd.DataFrame(static_network.degree)
        #print(degrees)

        time_covered = f.transmissions_time_covered
        
        transmission_array = f.model.transmissions_array
        batch_size = np.expand_dims(np.repeat(1, len(transmission_array), axis= -1), axis = -1)
        agg = np.concatenate([transmission_array, batch_size], axis = -1)
        transmissions_df = pd.DataFrame(agg, columns = ['t-1', 't', 'Source Node', 'Destination Node', 'Batch size'])

        outdir = f'/home/jacob/Documents/test/waxman_networks/{n_nodes}/WX{n}'
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        transmissions_df.to_csv(f"{outdir}/transmissions.txt", index = False, header = False)
        capacities.to_csv(f"{outdir}/capacity.txt", index = False, header = False)
        degrees.to_csv(f"{outdir}/degree.txt", index = False, header = False)

        transmissions_sara = transmissions_df[['Source Node', 'Destination Node', 't', 'Batch size']]
        transmissions_sara.to_csv(f"{outdir}/transmissions_sara.txt", index = False, header = False)


  if graph == "barabasi":
      #create a target function for each graph , store transmission network waxman
    for n in tqdm(range(number)):
    # at the beginning, call prepare() once:
        f.prepare(
        use_real_data = False, #False = synthetic data
        static_network = None, #use waxman graph
        n_nodes = n_nodes, #size of network
        max_t = 365, #time horizon
        expected_time_of_first_infection = 30, #30 days
        capacity_distribution = caps, #lambda size: np.ones(size), # any function accepting a 'size=' parameter
        delta_t_symptoms=60, #abort simulation after 60 days when symptoms first show up and testing becomes obsolet
        p_infection_by_transmission=0.5,
        pre_transmissions= None
        )

        capacities = pd.DataFrame(f.capacities)
        capacities = capacities.T

        time_covered = f.transmissions_time_covered
        transmissions = pd.DataFrame(f.transmissions_array)
        G = nx.from_pandas_edgelist(transmissions, source = 2, target = 3, edge_attr = None, create_using= nx.DiGraph)
        degrees = sorted(G.degree, key=lambda x: x[0], reverse=False)
        degrees = pd.DataFrame(degrees)

        transmission_array = f.model.transmissions_array
        batch_size = np.expand_dims(np.repeat(1, len(transmission_array), axis= -1), axis = -1)
        agg = np.concatenate([transmission_array, batch_size], axis = -1)
        transmissions_df = pd.DataFrame(agg, columns = ['t-1', 't', 'Source Node', 'Destination Node', 'Batch size'])
        
        outdir = f'/home/jacob/Documents/test/ba_networks/{n_nodes}/BA{n}'
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        transmissions_df.to_csv(f"{outdir}/transmissions.txt", index = False, header = False)
        capacities.to_csv(f"{outdir}/capacity.txt", index = False, header = False)
        degrees.to_csv(f"{outdir}/degree.txt", index = False, header = False)

        transmissions_sara = transmissions_df[['Source Node', 'Destination Node', 't', 'Batch size']]
        transmissions_sara.to_csv(f"{outdir}/transmissions_sara.txt", index = False, header = False)


"""create data"""
n_nodes = 57590 #1040 #120, 57590
create_graph(graph = "waxman", number = 10, n_nodes = n_nodes)
create_graph(graph = "barabasi", number = 10, n_nodes = n_nodes)

def evaluate_target_function_ba(f): 
  """load data to target function"""
  #create graph from .csv
  transmissions_ba = pd.read_csv(f"/home/jacob/Documents/test/ba_networks/{n_nodes}/BA0/transmissions.txt", header = None)
  capacities_ba = pd.read_csv(f"/home/jacob/Documents/test/ba_networks/{n_nodes}/BA0/capacity.txt", header = None)
  transmissions_ba = transmissions_ba.to_numpy()
  capacities_ba = capacities_ba.to_numpy().squeeze()

  f.prepare(
          use_real_data = False, #False = synthetic data
          static_network = None, #use barabasi albert
          n_nodes = n_nodes, #size of network
          max_t = 365, #time horizon
          expected_time_of_first_infection = 30, #30 days
          capacity_distribution = np.int(capacities_ba), #lambda size: np.ones(size), # any function accepting a 'size=' parameter
          delta_t_symptoms=60, #abort simulation after 60 days when symptoms first show up and testing becomes obsolet
          p_infection_by_transmission=0.5,
          pre_transmissions = transmissions_ba #use pre-stored transmission data
          )

  def n_infected_animals(is_infected):
      return np.sum(f.capacities * is_infected)

  def mean_square_and_stderr(n_infected_animals):
      values = n_infected_animals**2
      estimate = np.mean(values, axis=0)
      stderr = np.std(values, ddof=1, axis=0) / np.sqrt(values.shape[0])
      return estimate, stderr

  def est_prob_and_stderr(is_infected):
      ps = np.mean(is_infected, axis=0)
      stderrs = np.sqrt(ps * (1-ps) / is_infected.shape[0])
      return (ps, stderrs)

  n_simulations = 1000 
  num_cpu_cores = -1

  mean_sq_evaluation_params = { 
          'aggregation' : n_infected_animals,
          'statistic' : mean_square_and_stderr, 
          'n_simulations' : n_simulations, 
          'parallel': True,
          'num_cpu_cores': num_cpu_cores
          }
  
  x = np.repeat(1, n_nodes)
  y = f.evaluate(budget_allocation = x, n_simulations = n_simulations, aggregation=lambda a: a, statistic=est_prob_and_stderr)
  print(y)


def evaluate_target_function_wx(f): 
  """load data to target function"""
  #create graph from .csv
  transmissions_wx = pd.read_csv(f"/home/jacob/Documents/test/waxman_networks/{n_nodes}/WX0/transmissions.txt", header = None)
  capacities_wx = pd.read_csv(f"/home/jacob/Documents/test/waxman_networks/{n_nodes}/WX0/capacity.txt", header = None)
  capacities_wx = capacities_wx.to_numpy().squeeze()
  transmissions_wx = transmissions_wx.to_numpy()

  f.prepare(
          use_real_data = False, #False = synthetic data
          static_network = None, #use barabasi albert
          n_nodes = n_nodes, #size of network
          max_t = 365, #time horizon
          expected_time_of_first_infection = 30, #30 days
          capacity_distribution = np.int(capacities_wx), #lambda size: np.ones(size), # any function accepting a 'size=' parameter
          delta_t_symptoms=60, #abort simulation after 60 days when symptoms first show up and testing becomes obsolet
          p_infection_by_transmission=0.5,
          pre_transmissions = transmissions_wx #use pre-stored transmission data
          )

  def n_infected_animals(is_infected):
      return np.sum(f.capacities * is_infected)

  def mean_square_and_stderr(n_infected_animals):
      values = n_infected_animals**2
      estimate = np.mean(values, axis=0)
      stderr = np.std(values, ddof=1, axis=0) / np.sqrt(values.shape[0])
      return estimate, stderr

  def est_prob_and_stderr(is_infected):
      ps = np.mean(is_infected, axis=0)
      stderrs = np.sqrt(ps * (1-ps) / is_infected.shape[0])
      return (ps, stderrs)

  n_simulations = 1000 
  num_cpu_cores = -1

  mean_sq_evaluation_params = { 
          'aggregation' : n_infected_animals,
          'statistic' : mean_square_and_stderr, 
          'n_simulations' : n_simulations, 
          'parallel': True,
          'num_cpu_cores': num_cpu_cores
          }

  x = np.repeat(1, n_nodes)
  y = f.evaluate(budget_allocation = x, n_simulations = n_simulations, aggregation=lambda a: a, statistic=est_prob_and_stderr)
  print(y)

evaluate_target_function_ba(f)
evaluate_target_function_wx(f)

"""
Validation
transmission_array = f.model.transmissions_array
batch_size = np.expand_dims(np.repeat(1, len(transmission_array), axis= -1), axis = -1)
agg = np.concatenate([transmission_array, batch_size], axis = -1)
transmissions_df = pd.DataFrame(agg, columns = ['t-1', 't', 'Source Node', 'Destination Node', 'Batch size'])
transmissions_df.to_csv("/Users/admin/pyoptes/check.csv")
"""