'''
Pre-computes solutions for test budgets on the SI-simulation and saves the pairs on disk.
Pairs are computed for all network sizes and varying numbers of sentinels.
'''
import os.path

from pyoptes.optimization.budget_allocation import target_function as f

from pyoptes import map_low_dim_x_to_high_dim, create_test_strategy_prior, create_graph
from pyoptes import rms_tia, percentile_tia, mean_tia

import argparse
import numpy as np
from tqdm import tqdm
from time import time
import datetime

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--sentinels", type=int, default=1040,
                        help="Set the number of nodes that are used. Has to be smaller than or equal to n_nodes. "
                             "Default is 120 nodes.")
    parser.add_argument("--n_nodes", type=int, default=1040, choices=[120, 1040, 57590],
                        help="Si-simulation parameter. "
                             "Defines the number of nodes used by the SI-model to create a graph. "
                             "Default value is 120 nodes.")
    parser.add_argument('--n_runs', type=int, default=100,
                        help='The number of times the optimizer is run. Results are then averaged over all runs.'
                             'Default is 100 runs.')

    parser.add_argument("--statistic", choices=['mean', 'rms', '95perc'], default='rms',
                        help="Choose the statistic to be used by the target function. "
                             "Choose between mean, rms (root-mean-square) or 95perc (95th-percentile).")
    parser.add_argument("--n_simulations", type=int, default=10000,
                        help="Si-simulation parameter. Sets the number of runs the for the SI-model. "
                             "Higher values of n_simulations lower the variance of the output of the simulation. "
                             "Default value is 1000.")
    parser.add_argument('--graph_type', choices=['waxman', 'ba', 'syn'], default='syn',
                        help='Si-simulation parameter. Set the type of graph the simulation uses.'
                             ' Either Waxman,Synthetic or Barabasi-Albert (ba) can be used. Default is Synthetic.')
    parser.add_argument('--scale_total_budget', type=int, default=1, choices=[1, 4, 12],
                        help="SI-simulation parameter. Scales the total budget for SI-model. Default is 1.")

    parser.add_argument('--delta_t_symptoms', type=int, default=60,
                        help='Si-simulation parameter.. Sets the time (in days) after which an infection is detected'
                             ' automatically. Default is 60 days')
    parser.add_argument('--p_infection_by_transmission', type=float, default=0.5,
                        help='Si-simulation parameter. The probability of how likely a trade animal '
                             'infects other animals. Default is 0.5.')
    parser.add_argument('--expected_time_of_first_infection', type=int, default=30,
                        help='Si-simulation parameter. The expected time (in days) after which the first infection occurs. ')
    parser.add_argument('--parallel', type=bool, default=True,
                        help='Si-simulation parameter. Sets whether multiple simulations run are to be done in parallel'
                             'or sequentially. Default is set to parallel computation.')
    parser.add_argument("--num_cpu_cores", type=int, default=32,
                        help='Si-simulation parameter. Defines the number of cpus to be used for the simulation '
                             'parallelization. If more cpus are chosen than available, the max available are selected.'
                             '-1 selects all available cpus. Default are 32 cpus.')

    parser.add_argument('--path_data', default='../../data_pyoptes',
                        help="...")
    parser.add_argument('--path_networks', default='../../networks/data',
                        help='Location where the networks are saved to. '
                             'Path on cluster. /p/projects/ou/labs/gane/optes/mcmc_100nets/data'
                             '/p/projects/ou/labs/gane/optes/mcmc_100nets/data/')

    args = parser.parse_args()

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
                        path_experiment = os.path.join(args.path_data,
                                                       f'nodes_{n_nodes}',
                                                       f'sentinels_{sentinels}'
                                                       f'_budget_{scale_total_budget}N'
                                                       f'_graph_type_{graph_type}'
                                                       f'_statistic_{statistic}')
                        if not os.path.exists(path_experiment):
                            os.makedirs(path_experiment)

                        # define function to average the results of the simulation
                        if statistic == 'mean':
                            statistic = mean_tia
                        elif statistic == 'rms':
                            statistic = rms_tia
                        elif statistic == '95perc':
                            statistic = percentile_tia
                        else:
                            raise ValueError('Statistic not supported')

                        # The total budget is a function of the number of nodes in the network and
                        total_budget = scale_total_budget * n_nodes  # i.e., on average, nodes will do one test per year

                        list_all_prior_tf = []
                        list_all_prior_stderr = []

                        time_start = time()
                        for n in tqdm(range(args.n_runs), leave=False):
                            # print(f'Run {n + 1} of {args.n_runs},'
                            #       f' start time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

                            transmissions, capacities, degrees = create_graph(n, graph_type, n_nodes, args.path_networks)

                            f.prepare(n_nodes=n_nodes,
                                      capacity_distribution=capacities,
                                      pre_transmissions=transmissions,
                                      p_infection_by_transmission=args.p_infection_by_transmission,
                                      delta_t_symptoms=args.delta_t_symptoms,
                                      expected_time_of_first_infection=args.expected_time_of_first_infection,
                                      static_network=None,
                                      use_real_data=False)

                            # create a list of test strategies based on different heuristics
                            prior, prior_node_indices, prior_parameter = \
                                create_test_strategy_prior(n_nodes=n_nodes,
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

                                p = map_low_dim_x_to_high_dim(x=p,
                                                              number_of_nodes=n_nodes,
                                                              node_indices=prior_node_indices[i])

                                m, stderr = f.evaluate(budget_allocation=p,
                                                       n_simulations=args.n_simulations,
                                                       parallel=args.parallel,
                                                       num_cpu_cores=args.num_cpu_cores,
                                                       statistic=statistic)
                                list_prior_tf.append(m)
                                list_prior_stderr.append(stderr)
                            list_all_prior_tf.append(list_prior_tf)
                            list_all_prior_stderr.append(list_prior_stderr)

                        # save prior (and additional budgets in the form (budget, simulation-output)
                        # print('saving prior')
                        np.save(os.path.join(path_experiment, 'prior_tf.npy'), list_all_prior_tf)
                        np.save(os.path.join(path_experiment, 'prior_stderr.npy'), list_all_prior_stderr)

