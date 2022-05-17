import numpy as np
import pandas as pd
from pyoptes.optimization.budget_allocation import target_function as f
from target_function_parameter import all_parameters_aggregation as tf_aggregation
from target_function_parameter import all_parameters_statistics as tf_statistics


def compute_baseline(evaluation_params, n_inputs, total_budget, f, degree, network, number, budget):
    capacities = f.capacities
    capacities_per_node = []
    for i in range(len(capacities)):
        capacities_per_node.append((i, capacities[i]))
    def Extract(lst):
        return list(list(zip(*lst))[0])
    capacity_values = sorted(capacities_per_node, key=lambda x: x[1], reverse=True)
    capacity = Extract(capacity_values)

    if n_inputs == 120:
        num = list(np.arange(start=113, stop=1020))
        unwanted_num = num
    elif n_inputs  == 1040:
        num = list(np.arange(start=955, stop=1040))
        unwanted_num = num
    elif n_inputs  == 57590:
        num = list(np.arange(start=54135, stop=57590))
        unwanted_num = num


    capacity = [ele for ele in capacity if ele not in unwanted_num]

    n_2 = int(np.round(n_inputs/2, decimals=0))
    n_6 = int(np.round(n_inputs/6, decimals=0))
    n_12 = int(np.round(n_inputs/12, decimals=0))
    n_26 = int(np.round(n_inputs/26, decimals=0))
    n_52 = int(np.round(n_inputs/52, decimals=0))
    
    """uniform distribution among N/2 highest degree nodes"""
    highest_degrees = degree[:n_2]
    sentinels = highest_degrees
    weights = np.zeros(n_inputs)
    weights[sentinels] = 1
    shares = weights / weights.sum()
    budget_hd = shares * total_budget
    budget_hd = budget * budget_hd
    ratio_infected, node_metrics, mean_sq_metrics, mean_metrics, p95_metrics = f.evaluate(budget_hd, **evaluation_params)
    metrics = np.concatenate((node_metrics, ratio_infected, mean_sq_metrics, mean_metrics, p95_metrics), axis = None)
    metrics_hd2 = metrics.reshape(1, len(metrics))


    """uniform distribution among N/2 highest capacities nodes"""
    highest_capacaities = capacity[:n_2]
    sentinels = highest_capacaities
    weights = np.zeros(n_inputs)
    weights[sentinels] = 1
    shares = weights / weights.sum()
    budget_hd = shares * total_budget
    budget_hd = budget * budget_hd

    ratio_infected, node_metrics, mean_sq_metrics, mean_metrics, p95_metrics = f.evaluate(budget_hd, **evaluation_params)
    metrics = np.concatenate((node_metrics, ratio_infected, mean_sq_metrics, mean_metrics, p95_metrics), axis = None)
    metrics_hc2 = metrics.reshape(1, len(metrics))

    """uniform distribution among N/6 highest degree nodes"""
    highest_degrees = degree[:n_6]
    sentinels = highest_degrees
    weights = np.zeros(n_inputs)
    weights[sentinels] = 1
    shares = weights / weights.sum()
    budget_hd = shares * total_budget
    budget_hd = budget * budget_hd
    ratio_infected, node_metrics, mean_sq_metrics, mean_metrics, p95_metrics = f.evaluate(budget_hd, **evaluation_params)
    metrics = np.concatenate((node_metrics, ratio_infected, mean_sq_metrics, mean_metrics, p95_metrics), axis = None)
    metrics_hd6 = metrics.reshape(1, len(metrics))


    """uniform distribution among N/6 highest capacities nodes"""
    highest_capacaities = capacity[:n_6]
    sentinels = highest_capacaities
    weights = np.zeros(n_inputs)
    weights[sentinels] = 1
    shares = weights / weights.sum()
    budget_hd = shares * total_budget
    budget_hd = budget * budget_hd
    ratio_infected, node_metrics, mean_sq_metrics, mean_metrics, p95_metrics = f.evaluate(budget_hd, **evaluation_params)
    metrics = np.concatenate((node_metrics, ratio_infected, mean_sq_metrics, mean_metrics, p95_metrics), axis = None)
    metrics_hc6 = metrics.reshape(1, len(metrics))


    """uniform distribution among N/12 highest degree nodes"""
    highest_degrees = degree[:n_12]
    sentinels = highest_degrees
    weights = np.zeros(n_inputs)
    weights[sentinels] = 1
    shares = weights / weights.sum()
    budget_hd = shares * total_budget
    budget_hd = budget * budget_hd
    ratio_infected, node_metrics, mean_sq_metrics, mean_metrics, p95_metrics = f.evaluate(budget_hd, **evaluation_params)
    metrics = np.concatenate((node_metrics, ratio_infected, mean_sq_metrics, mean_metrics, p95_metrics), axis = None)
    metrics_hd12 = metrics.reshape(1, len(metrics))


    """uniform distribution among N/12 highest capacities nodes"""
    highest_capacaities = capacity[:n_12]
    sentinels = highest_capacaities
    weights = np.zeros(n_inputs)
    weights[sentinels] = 1
    shares = weights / weights.sum()
    budget_hd = shares * total_budget
    budget_hd = budget * budget_hd
    ratio_infected, node_metrics, mean_sq_metrics, mean_metrics, p95_metrics = f.evaluate(budget_hd, **evaluation_params)
    metrics = np.concatenate((node_metrics, ratio_infected, mean_sq_metrics, mean_metrics, p95_metrics), axis = None)
    metrics_hc12 = metrics.reshape(1, len(metrics))
    

    """uniform distribution among N/26 highest degree nodes"""
    highest_degrees = degree[:n_26]
    sentinels = highest_degrees
    weights = np.zeros(n_inputs)
    weights[sentinels] = 1
    shares = weights / weights.sum()
    budget_hd = shares * total_budget
    budget_hd = budget * budget_hd
    ratio_infected, node_metrics, mean_sq_metrics, mean_metrics, p95_metrics = f.evaluate(budget_hd, **evaluation_params)
    metrics = np.concatenate((node_metrics, ratio_infected, mean_sq_metrics, mean_metrics, p95_metrics), axis = None)
    metrics_hd26 = metrics.reshape(1, len(metrics))
    

    """uniform distribution among N/26 highest capacities nodes"""
    highest_capacaities = capacity[:n_26]
    sentinels = highest_capacaities
    weights = np.zeros(n_inputs)
    weights[sentinels] = 1
    shares = weights / weights.sum()
    budget_hd = shares * total_budget
    budget_hd = budget * budget_hd
    ratio_infected, node_metrics, mean_sq_metrics, mean_metrics, p95_metrics = f.evaluate(budget_hd, **evaluation_params)
    metrics = np.concatenate((node_metrics, ratio_infected, mean_sq_metrics, mean_metrics, p95_metrics), axis = None)
    metrics_hc26 = metrics.reshape(1, len(metrics))


    """uniform distribution among N/52 highest degree nodes"""
    highest_degrees = degree[:n_52]
    sentinels = highest_degrees
    weights = np.zeros(n_inputs)
    weights[sentinels] = 1
    shares = weights / weights.sum()
    budget_hd = shares * total_budget
    budget_hd = budget * budget_hd

    ratio_infected, node_metrics, mean_sq_metrics, mean_metrics, p95_metrics = f.evaluate(budget_hd, **evaluation_params)
    metrics = np.concatenate((node_metrics, ratio_infected, mean_sq_metrics, mean_metrics, p95_metrics), axis = None)
    metrics_hd52 = metrics.reshape(1, len(metrics))


    """uniform distribution among N/52 highest capacity nodes"""
    highest_capacaities = capacity[:n_52]
    sentinels = highest_capacaities
    weights = np.zeros(n_inputs)
    weights[sentinels] = 1
    shares = weights / weights.sum()
    budget_hd = shares * total_budget
    budget_hd = budget * budget_hd

    ratio_infected, node_metrics, mean_sq_metrics, mean_metrics, p95_metrics = f.evaluate(budget_hd, **evaluation_params)
    metrics = np.concatenate((node_metrics, ratio_infected, mean_sq_metrics, mean_metrics, p95_metrics), axis = None)
    metrics_hc52 = metrics.reshape(1, len(metrics))

    """ do the same for an x that is based on the total number of incoming transmissions per node:"""
    target_list = f.model.transmissions_array[:, 3]
    values, counts = np.unique(target_list, return_counts=True)
    weights = np.zeros(n_inputs)
    weights[values] = counts
    shares = weights / weights.sum()
    budget_hd = shares * total_budget
    budget_hd = budget * budget_hd
    ratio_infected, node_metrics, mean_sq_metrics, mean_metrics, p95_metrics = f.evaluate(budget_hd, **evaluation_params)
    metrics = np.concatenate((node_metrics, ratio_infected, mean_sq_metrics, mean_metrics, p95_metrics), axis = None)
    metrics_trans = metrics.reshape(1, len(metrics))

    """baseline budget distribution among all nodes = testing each node once p.a"""
    test_1_per_year = np.ones_like(budget_hd)
    test_1_per_year = budget * test_1_per_year
    ratio_infected, node_metrics, mean_sq_metrics, mean_metrics, p95_metrics = f.evaluate(test_1_per_year, **evaluation_params)
    metrics = np.concatenate((node_metrics, ratio_infected, mean_sq_metrics, mean_metrics, p95_metrics), axis = None)
    metrics_once = metrics.reshape(1, len(metrics))

    """zero budget"""
    zero_budget = np.zeros_like(budget_hd)
    zero_budget = budget * zero_budget
    ratio_infected, node_metrics, mean_sq_metrics, mean_metrics, p95_metrics = f.evaluate(zero_budget, **evaluation_params)
    metrics = np.concatenate((node_metrics, ratio_infected, mean_sq_metrics, mean_metrics, p95_metrics), axis = None)
    metrics_zero = metrics.reshape(1,len(metrics))

    metrics_matrix = np.concatenate((metrics_hd2, metrics_hc2, metrics_hd6, metrics_hc6, metrics_hd12, metrics_hc12, metrics_hd26, metrics_hc26, metrics_hd52, metrics_hc52, metrics_trans, metrics_once, metrics_zero), axis = 0)
    df = pd.DataFrame(metrics_matrix)
    df.to_csv(f"/Users/admin/pyoptes/src/pyoptes/optimization/budget_allocation/supervised_learning/baselines/{network}/{n_inputs}/{network}_{number}_{n_inputs}_{budget}N.csv", header = None, index=False)


def evaluate_baselines(n_nodes, networks, evaluation_params, budget, dir):
    for i in range(2):
        network = networks[i]
        net = dir[0]
        for n in range(2):
            nodes = n_nodes[n]
            for b in range(len(budget)):
                budgets = budget[b]
                for j in range(100):                    

                    transmissions = pd.read_csv(f"/Users/admin/pyoptes/src/pyoptes/optimization/budget_allocation/supervised_learning/{network}/{nodes}/{net}{j}/dataset.txt", header = None)
                    transmissions = transmissions[[2, 2, 0, 1, 3]]  
                    transmissions = transmissions.to_numpy()

                    capacities = pd.read_csv(f"/Users/admin/pyoptes/src/pyoptes/optimization/budget_allocation/supervised_learning/{network}/{nodes}/{net}{j}/barn_size.txt", header = None)
                    capacities = capacities.iloc[0][:nodes].to_numpy() #delete last nan entry
                    degrees = pd.read_csv(f"/Users/admin/pyoptes/src/pyoptes/optimization/budget_allocation/supervised_learning/{network}/{nodes}/{net}{j}/degree_sentil.txt", header = None)
                    degrees = degrees.iloc[0][:-1].to_numpy(dtype=np.int64) #delete last nan entry

                    # at the beginning, call prepare() once:
                    f.prepare(
                        use_real_data = False, #False = synthetic data
                        static_network = None, #use waxman graph
                        n_nodes = nodes, #size of network
                        max_t = 365, #time horizon
                        expected_time_of_first_infection = 30, #30 days
                        capacity_distribution = capacities, #lambda size: np.ones(size), # any function accepting a 'size=' parameter
                        delta_t_symptoms=60, #abort simulation after 60 days when symptoms first show up and testing becomes obsolet
                        p_infection_by_transmission=0.5,
                        pre_transmissions = transmissions #use pre-stored transmission data
                        )

                    compute_baseline(evaluation_params = evaluation_params, n_inputs=nodes, total_budget=nodes, f=f, degree=degrees, network = network, number = j, budget=budgets)


budget = [1.0, 4.0, 12.0]
n_nodes = [120, 1040]
networks = ["synthetic_networks", "waxman", "barabasi"] #focus on syn_net
dir = ["syndata", "WX", "BA"]  #focus on syn_net
n_simulations = 100000
num_cpu_cores = -1
all_params = { 
        'aggregation' : tf_aggregation,
        'statistic' : tf_statistics,
        'n_simulations' : n_simulations, 
        'parallel': True,
        'num_cpu_cores': num_cpu_cores
        }

evaluate_baselines(n_nodes=n_nodes, networks=networks, evaluation_params=all_params, budget = budget, dir = dir)