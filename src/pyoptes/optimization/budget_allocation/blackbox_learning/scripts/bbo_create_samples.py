'''
Pre-computes solutions for test budgets on the SI-simulation and saves the pairs on disk.
Pairs are computed for all network sizes and varying numbers of sentinels.
'''
import os.path

from pyoptes.optimization.budget_allocation import target_function as f

from pyoptes import bo_map_low_dim_x_to_high_dim, bo_create_test_strategy_prior, bo_create_graph
from pyoptes import bo_rms_tia, bo_percentile_tia, bo_mean_tia

import argparse
import numpy as np
from tqdm import tqdm
from time import time


def bbo_create_samples(sentinels: int = 1040,
                       n_nodes: int = 1040,
                       n_runs: int = 100,
                       statistic_str: str = "mean",
                       n_simulations: int = 1000,
                       graph_type: str = "syn",
                       scale_total_budget: int = 1,
                       parallel: bool = True,
                       num_cpu_cores: int = 32,
                       delta_t_symptoms: int = 60,
                       p_infection_by_transmission: float = 0.5,
                       expected_time_of_first_infection: int = 30,
                       path_data: str = "../../data_pyoptes/",
                       path_networks: str = "../../networks/data"
):

    list_nodes = [120, 1040, 57590]
    list_sentinels = [0.03, 0.06, 1]
    list_budget_scale = [1, 4, 12]
    list_graph_type = ['syn'] #['waxman', 'ba', 'syn'] # how do I deal with the fact that the graphs are not all at the same location ??
    list_statistics = ['mean', 'rms', '95perc']
    for n_nodes in tqdm(list_nodes):
        for sentinels_scale in list_sentinels:
            for scale_total_budget in list_budget_scale:
                for graph_type in list_graph_type:
                    for statistic in list_statistics:

                        sentinels = int(n_nodes*sentinels_scale)
                        # prepare the directory for the plots, hyperparameters and results
                        path_experiment = os.path.join(path_data,
                                                       f'nodes_{n_nodes}',
                                                       f'sentinels_{sentinels}'
                                                       f'_budget_{scale_total_budget}N'
                                                       f'_graph_type_{graph_type}'
                                                       f'_statistic_{statistic}')
                        if not os.path.exists(path_experiment):
                            os.makedirs(path_experiment)

                        # define function to average the results of the simulation
                        if statistic_str == 'mean':
                            statistic = bo_mean_tia
                        elif statistic_str == 'rms':
                            statistic = bo_rms_tia
                        elif statistic_str == '95perc':
                            statistic = bo_percentile_tia
                        else:
                            raise ValueError('Statistic not supported')

                        # The total budget is a function of the number of nodes in the network and
                        total_budget = scale_total_budget * n_nodes  # i.e., on average, nodes will do one test per year

                        list_all_prior_tf = []
                        list_all_prior_stderr = []

                        time_start = time()
                        for n in tqdm(range(n_runs), leave=False):
                            # print(f'Run {n + 1} of {n_runs},'
                            #       f' start time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

                            transmissions, capacities, degrees = bo_create_graph(n, graph_type, n_nodes, path_networks)

                            f.prepare(n_nodes=n_nodes,
                                      capacity_distribution=capacities,
                                      pre_transmissions=transmissions,
                                      p_infection_by_transmission=p_infection_by_transmission,
                                      delta_t_symptoms=delta_t_symptoms,
                                      expected_time_of_first_infection=expected_time_of_first_infection,
                                      static_network=None,
                                      use_real_data=False)

                            # create a list of test strategies based on different heuristics
                            prior, prior_node_indices, prior_parameter = \
                                bo_create_test_strategy_prior(n_nodes=n_nodes,
                                                              node_degrees=degrees,
                                                              node_capacities=capacities,
                                                              total_budget=total_budget,
                                                              sentinels=sentinels,
                                                              mixed_strategies=False,
                                                              only_baseline=False)

                            # evaluate the strategies in the prior
                            list_prior_tf = []
                            list_prior_stderr = []

                            for i, p in tqdm(enumerate(prior), leave=False, total=len(prior)):

                                p = bo_map_low_dim_x_to_high_dim(x=p,
                                                                 number_of_nodes=n_nodes,
                                                                 node_indices=prior_node_indices[i])

                                m, stderr = f.evaluate(budget_allocation=p,
                                                       n_simulations=n_simulations,
                                                       parallel=parallel,
                                                       num_cpu_cores=num_cpu_cores,
                                                       statistic=statistic)
                                list_prior_tf.append(m)
                                list_prior_stderr.append(stderr)
                            list_all_prior_tf.append(list_prior_tf)
                            list_all_prior_stderr.append(list_prior_stderr)

                        # save prior (and additional budgets in the form (budget, simulation-output)
                        # print('saving prior')
                        np.save(os.path.join(path_experiment, 'prior_tf.npy'), list_all_prior_tf)
                        np.save(os.path.join(path_experiment, 'prior_stderr.npy'), list_all_prior_stderr)

