"""
Simple demonstration to illustrate how to generate training samples
"""

import numpy as np
from pyoptes import set_seed
from pyoptes.optimization.budget_allocation import target_function as f
import warnings
import networkx as nx
from scipy.stats import lognorm
from tqdm import tqdm
import pylab as plt


warnings.filterwarnings("ignore")

# set some seed to get reproducible results:
set_seed(1)

# define size of our network
n_nodes = 120

# define distribution of our node test capacities 
def caps(size): 
  return lognorm.rvs(s=2, scale=np.exp(4), size=size)

# generate a 11 by 11 2d lattice with nodes numbered 0 to 120:
#lattice = nx.DiGraph(nx.to_numpy_array(nx.lattice.grid_2d_graph(11, 11)))

graph = 'waxman'

# generate a Waxman graph:
waxman = nx.waxman_graph(n_nodes)
pos = dict(waxman.nodes.data('pos'))
# convert into a directed graph:
static_network = nx.DiGraph(nx.to_numpy_array(waxman))

# at the beginning, call prepare() once:
f.prepare(
  use_real_data = False, #False = synthetic data
  static_network = None, #use waxman graph
  n_nodes = n_nodes, #size of network
  max_t = 365, #time horizon
  expected_time_of_first_infection = 30, #30 days
  capacity_distribution = caps, #lambda size: np.ones(size), # any function accepting a 'size=' parameter
  delta_t_symptoms=60, #abort simulation after 60 days when symptoms first show up and testing becomes obsolet
  p_infection_by_transmission=0.5 
  )

n_inputs = f.get_n_inputs() #number of nodes of our network

print("n_inputs (=number of network nodes):", n_inputs)


times_budget = [1.0]#, 4.0, 12.0]
total_budget =  [times_budget[0] * n_inputs]#, times_budget[1] * n_inputs, times_budget[2] * n_inputs] #total budget equals number of nodes
n_samples = 100 #how many samples we want to generate
n_simulations = 10000 #run n_simulations of our target function to reduce std error
num_cpu_cores = -1 #use all cpu cores


def est_prob_and_stderr(is_infected):
    ps = np.mean(is_infected, axis=0)
    stderrs = np.sqrt(ps * (1-ps) / is_infected.shape[0])
    return (ps, stderrs)

def n_infected_animals(is_infected):
    return np.sum(f.capacities * is_infected)

def mean_square_and_stderr(n_infected_animals):
    values = n_infected_animals**2
    estimate = np.mean(values, axis=0)
    stderr = np.std(values, ddof=1, axis=0) / np.sqrt(values.shape[0])
    return estimate, stderr


#evaluation params of our target function

#generate samples: 
  #mean_square_and_stderr = np.mean(a**2)
  #np.mean
  #lambda a: np.percentile(a, 95)
  #share detected 

mean_sq_evaluation_params = { 
        'aggregation' : n_infected_animals,
        'statistic' : mean_square_and_stderr, 
        'n_simulations' : n_simulations, 
        'parallel': True,
        'num_cpu_cores': num_cpu_cores
        }

p95_evaluation_params = { 
        'aggregation' : n_infected_animals,
        'statistic' : lambda a: np.percentile(a, 95), 
        'n_simulations' : n_simulations, 
        'parallel': True,
        'num_cpu_cores': num_cpu_cores
        }

mean_evaluation_params = { 
        'aggregation' : n_infected_animals,
        'statistic' : np.mean, 
        'n_simulations' : n_simulations, 
        'parallel': True,
        'num_cpu_cores': num_cpu_cores
        }

share_detected_evaluation_params = { 
        'aggregation' : n_infected_animals,
        'statistic' : mean_square_and_stderr, 
        'n_simulations' : n_simulations, 
        'parallel': True,
        'num_cpu_cores': num_cpu_cores
        }

samples_per_iteration = 3
opt_target_function = "opt_target_function" #RMS_TIA, 95_TIA, share_detected
list_baseline = []

# generate n_samples
for j in tqdm(range(n_samples)):
    f.prepare(
        use_real_data=False,  # False = synthetic data
        static_network=static_network,  # use waxman graph
        n_nodes=n_nodes,  # size of network
        max_t=365,  # time horizon
        expected_time_of_first_infection=30,  # 30 days
        capacity_distribution=caps,  # lambda size: np.ones(size), # any function accepting a 'size=' parameter
        delta_t_symptoms=60,  # abort simulation after 60 days when symptoms first show up and testing becomes obsolet
        p_infection_by_transmission=0.5
    )
    #x that is based on sentinels
    no_sent = np.random.choice(np.arange(0, n_inputs), 1) #choose a number no_sent between 1 and number of nodes
    sentinels = list(np.random.choice(np.arange(0, n_inputs), no_sent, replace=False)) #decide which nodes are sentinels
    weights = np.zeros(n_inputs) 
    weights[sentinels] = 1 
    shares_sent = weights / weights.sum()

    #x that is randomly sampled
    weights = np.random.rand(n_inputs)
    shares_rnd = weights / weights.sum()

    # x that is exp. randomly sampled
    weights = np.random.exponential(size = n_inputs)
    shares_exp = weights / weights.sum()

    # print('-------------------------------------------------------')
    x_sent = shares_sent * total_budget[0] #distribute budget among sentinels
    # prob_node_sent, std_errs_node_sent = f.evaluate(x_sent, n_simulations=n_simulations, aggregation=lambda a: a, statistic=est_prob_and_stderr)
    (dmg_total_sent, se_total_sent) = f.evaluate(x_sent, **mean_sq_evaluation_params)
    # dmg_sent = np.concatenate((prob_node_sent, dmg_total_sent), axis = None)
    # se_sent = np.concatenate((std_errs_node_sent, se_total_sent), axis = None)
    list_baseline.append([dmg_total_sent, se_total_sent])
    # print(f'std_err prob infection per node sentinels: {np.sum(std_errs_node_sent)/n_nodes}')
    # print(f'std_err mean sq total damage sentinel: {se_total_sent/dmg_total_sent}')

m = np.mean(list_baseline, axis=0)
print(f'Mean damage sentinel: {m}, {m[1]/m[0]}')

m = np.array(list_baseline)[:, 0]
su = [m[i] - list_baseline[i][1] for i in range(len(m))]
sd = [m[i] + list_baseline[i][1] for i in range(len(m))]

average = np.ones(len(m)) * m[0]
plt.plot(range(len(list_baseline)), m, label='baseline')
plt.plot(range(len(list_baseline)), average, label='average baseline')
# add standard error of the baseline
plt.plot(range(len(list_baseline)), su,
         label='stderr baseline', linestyle='dotted', color='red')
plt.plot(range(len(list_baseline)), sd,
         linestyle='dotted', color='red')
plt.title(f'Baseline mean & stderr over {n_samples} synthetic networks')
plt.xlabel('Iteration')
plt.ylabel('SI-model output')
plt.legend()
plt.show()

      #x that is randomly sampled
      # x_rnd = shares_rnd * total_budget[i]
      # prob_node_rnd, std_errs_node_rnd = f.evaluate(x_rnd, n_simulations=n_simulations, aggregation=lambda a: a, statistic=est_prob_and_stderr)
      # (dmg_total_rnd, se_total_rnd) = f.evaluate(x_rnd, **mean_sq_evaluation_params)

      # dmg_rnd = np.concatenate((prob_node_rnd, dmg_total_rnd), axis = None)
      # se_rnd = np.concatenate((std_errs_node_rnd, se_total_rnd), axis = None)
      
      # print(f'std_err prob infection per node randomly sampled budget: {np.sum(std_errs_node_rnd)/n_nodes}')
      # print(f'std_err mean sq total damage randomly sampled budget: {se_total_rnd/dmg_total_rnd}')
      
      # x that is exp. randomly sampled
      # x_exp = shares_exp * total_budget[i]

      # prob_node_exp, std_errs_node_exp = f.evaluate(x_exp, n_simulations=n_simulations, aggregation=lambda a: a, statistic=est_prob_and_stderr)
      # (dmg_total_exp, se_total_exp) = f.evaluate(x_exp, **mean_sq_evaluation_params)
      
      # dmg_exp = np.concatenate((prob_node_exp, dmg_total_exp), axis = None)
      # se_exp = np.concatenate((std_errs_node_exp, se_total_exp), axis = None)
      
      # print(f'std_err prob infection per node exp randomly sampled budget: {np.sum(std_errs_node_exp)/n_nodes}')
      # print(f'std_err mean sq total damage exp randomly sampled budget: {se_total_exp/dmg_total_exp}')

"""
      #write our inputs to csv
      f_1 = open (f'/Users/admin/pyoptes/src/pyoptes/optimization/budget_allocation/supervised_learning/data_per_node/{graph}_inputs_{times_budget[i]}xbudget_{opt_target_function}.csv', 'a')
      writer = csv.writer(f_1)
      writer.writerow(x_rnd)
      writer.writerow(x_exp)
      writer.writerow(x_sent)
      f_1.close()

      #write our targets to csv
      f_2 = open (f'/Users/admin/pyoptes/src/pyoptes/optimization/budget_allocation/supervised_learning/data_per_node/{graph}_targets_{times_budget[i]}xbudget_{opt_target_function}.csv', 'a')
      writer = csv.writer(f_2)
      writer.writerow(dmg_rnd)
      writer.writerow(dmg_exp)
      writer.writerow(dmg_sent)
      #f_2.write(np.str(dmg_node_rnd) + "\n" + np.str(dmg_node_exp) + "\n" + np.str(dmg_node_sent) + "\n")
      f_2.close()

      #write our standard errors to csv (optional)
      f_3 = open (f'/Users/admin/pyoptes/src/pyoptes/optimization/budget_allocation/supervised_learning/data_per_node/{graph}_standard_errors_{times_budget[i]}xbudget_{opt_target_function}.csv', 'a')
      writer = csv.writer(f_3)
      writer.writerow(se_rnd)
      writer.writerow(se_exp)
      writer.writerow(se_sent)
      #f_3.write(np.str(se_node_rnd) + "\n" + np.str(se_node_exp) + "\n" + np.str(se_node_sent) + "\n")
      f_3.close()

      print(f'\ngenerated: {(j+1)*samples_per_iteration} pairs of datapoints x and y')
"""