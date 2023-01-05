'''
Computes the average target function output for the uniform baseline.
'''


from pyoptes.optimization.budget_allocation import target_function as f
from pyoptes import bo_create_graph, bo_compute_average_otf_and_stderr

import numpy as np
from tqdm import tqdm
from scipy.stats.mstats import mjci


def rms_tia(n_infected_animals):
    values = n_infected_animals**2
    estimate = np.sqrt(np.mean(values, axis=0))
    stderr = np.std(values, ddof=1, axis=0) / np.sqrt(values.shape[0])
    stderr = stderr/(2*estimate)
    return estimate, stderr


def mean_tia(n_infected_animals):
    estimate = np.mean(n_infected_animals, axis=0)
    stderr = np.std(n_infected_animals, ddof=1, axis=0) / np.sqrt(n_infected_animals.shape[0])
    return estimate, stderr


def percentile_tia(n_infected_animals):
    estimate = np.percentile(n_infected_animals, 95, axis=0)
    stderr = mjci(n_infected_animals, prob=[0.95], axis=0)[0]
    return estimate, stderr


def bbo_sanity_check(n_nodes,
                     n_runs,
                     graph_type,
                     scale_total_budget,
                     path_networks,):

    total_budget = scale_total_budget * n_nodes

    list_baseline_otf = []  # baseline  function value on each network and corresponding standard error
    list_baseline_otf_stderr = []

    for n in tqdm(range(n_runs)):
        transmissions, capacities, degrees = bo_create_graph(n, graph_type, n_nodes, path_networks)

        f.prepare(n_nodes=n_nodes,
                  capacity_distribution=capacities,
                  pre_transmissions=transmissions,
                  p_infection_by_transmission=0.5,
                  delta_t_symptoms=60,
                  expected_time_of_first_infection=30,
                  static_network=None,
                  use_real_data=False)

        x_baseline = np.array([total_budget / n_nodes for _ in range(n_nodes)])

        m, stderr = f.evaluate(x_baseline,
                               n_simulations=10000,
                               parallel=True,
                               num_cpu_cores=32,
                               statistic=rms_tia)

        list_baseline_otf.append(m)
        list_baseline_otf_stderr.append(stderr)
    print(list_baseline_otf)
    average_baseline, average_baseline_stderr = bo_compute_average_otf_and_stderr(list_baseline_otf,
                                                                                  list_baseline_otf_stderr,
                                                                                  n_runs=n_runs)

    print('average_baseline', average_baseline)
    print('average_baseline_stderr', average_baseline_stderr)