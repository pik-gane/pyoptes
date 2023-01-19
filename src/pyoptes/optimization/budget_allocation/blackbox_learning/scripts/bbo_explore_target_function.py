'''
Compute the target function value for different number of sentinels (either highest degree, transmissions, capacities)

'''

from pyoptes.optimization.budget_allocation import target_function as f
from pyoptes import bo_create_graph, bo_compute_average_otf_and_stderr, bo_map_low_dim_x_to_high_dim
from pyoptes import bo_choose_sentinels
from pyoptes import bo_rms_tia, bo_percentile_tia, bo_mean_tia
from pyoptes import bo_plot_effect_of_different_sentinels

import numpy as np
from tqdm import tqdm


def bbo_explore_target_function(n_runs: int = 100,
                                statistic_str: str = "mean",
                                n_simulations: int = 1000,
                                graph_type: str = "syn",
                                scale_total_budget: int = 1,
                                parallel: bool = True,
                                num_cpu_cores: int = 32,
                                delta_t_symptoms: int = 60,
                                p_infection_by_transmission: float = 0.5,
                                expected_time_of_first_infection: int = 30,
                                mode_choose_sentinels: str = "degree",
                                path_networks: str = "../../networks/data",
                                path_plot: str = '../data/blackbox_learning/results/',
                                step_size: int = 5):

    nodes = [120, 1040, 57590]

    # define function to average the results of the simulation
    if statistic_str == 'mean':
        statistic = bo_mean_tia
    elif statistic_str == 'rms':
        statistic = bo_rms_tia
    elif statistic_str == '95perc':
        statistic = bo_percentile_tia
    else:
        raise ValueError('Statistic not supported')

    for n in nodes:
        print(f'Running simulation with {n} nodes')
        # create a list of sentinels from 0 to all sentinels
        sentinels = list(range(0, n+step_size, step_size))

        total_budget = n

        list_all_m = []
        list_all_stderr = []

        # compute the target function value for different number of sentinels
        for s in tqdm(sentinels):
            list_m = []
            list_stderr = []
            for run in tqdm(range(n_runs), leave=False):

                # create graph and prepare the simulation
                transmissions, capacities, degrees = bo_create_graph(n=run,
                                                                     graph_type=graph_type,
                                                                     n_nodes=n,
                                                                     base_path=path_networks)

                f.prepare(n_nodes=n,
                          capacity_distribution=capacities,
                          pre_transmissions=transmissions,
                          p_infection_by_transmission=p_infection_by_transmission,
                          delta_t_symptoms=delta_t_symptoms,
                          expected_time_of_first_infection=expected_time_of_first_infection,
                          static_network=None,
                          use_real_data=False)

                sentinel_indices = bo_choose_sentinels([degrees, None, None], s, mode_choose_sentinels)

                # create a budget vector
                budget = np.ones(s)*total_budget/s

                budget = bo_map_low_dim_x_to_high_dim(budget, n, sentinel_indices)

                m, stderr = f.evaluate(budget_allocation=budget,
                                       n_simulations=n_simulations,
                                       parallel=parallel,
                                       num_cpu_cores=num_cpu_cores,
                                       statistic=statistic)

                list_m.append(m)
                list_stderr.append(stderr)
            # average run results
            mm, m_stderr = bo_compute_average_otf_and_stderr(list_m, list_stderr, n_runs)
            # save the averaged results
            list_all_m.append(mm)
            list_all_stderr.append(m_stderr)
        # create a bar plot from mean, stderr pair for each sentinel number
        # x-axis is number of sentinels, y are mean and stderr

        index_minimum = np.argmin(list_all_m)
        minimum = [sentinels[index_minimum], list_all_m[index_minimum]]

        # TODO save the raw-data of the stuff

        # TODO maybe create additional plots with capacity sentinels and other attributes
        bo_plot_effect_of_different_sentinels(number_of_sentinels=sentinels,
                                              m=list_all_m,
                                              stderr=list_all_stderr,
                                              path_experiment=path_plot,
                                              title=f'Simulations with increasing number of sentinels. {n} nodes',
                                              n_nodes=n,
                                              mode_choose_sentinels=mode_choose_sentinels,
                                              minimum=minimum,
                                              step_size=step_size)
