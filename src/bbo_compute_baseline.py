'''
Compute baseline values with the
'''

import os.path

from pyoptes.optimization.budget_allocation import target_function as f

from pyoptes import bo_cma, bo_pyGPGO

from pyoptes import choose_sentinels, baseline
from pyoptes import map_low_dim_x_to_high_dim, create_test_strategy_prior
from pyoptes import save_hyperparameters, save_results, plot_prior, create_graph, save_raw_data
from pyoptes import plot_time_for_optimization, plot_optimizer_history, evaluate_prior
from pyoptes import compute_average_otf_and_stderr, softmax

import argparse
import numpy as np
from tqdm import tqdm
from scipy.stats.mstats import mjci
from time import time
import datetime


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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--sentinels", type=int, default=1040,
                        help="Set the number of nodes that are used. Has to be smaller than or equal to n_nodes. "
                             "Default is 120 nodes.")

    parser.add_argument('--use_prior', type=bool, default=True,
                        help='GPGO optimizer parameter. Sets whether the surrogate function is fitted with priors '
                             'created by heuristics or by sampling random point. Only works when n_nodes and sentinels'
                             'are the same size. Default is True.')
    parser.add_argument('--prior_mixed_strategies', type=bool, default=False,
                        help='GPGO optimizer parameter. '
                             'Sets whether to use test strategies that mix highest degrees and capacities in the prior.'
                             'If set to no the prior has the same shape for all network sizes.')
    parser.add_argument('--prior_only_baseline', type=bool, default=False,
                        help='GPGO optimizer parameter. Sets whether to use only the baseline strategy in the prior.'
                             'If true the prior consists of only one item.')

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

    parser.add_argument('--mode_choose_sentinels', choices=['degree', 'capacity', 'transmission'], default='degree',
                        help='Sets the mode of how sentinels are chosen. ')
    parser.add_argument('--save_test_strategies', type=bool, default='',
                        help='Sets whether to save the test strategies that are evaluate in the optimization.')
    parser.add_argument('--plot_prior', type=bool, default='',
                        help='')
    parser.add_argument("--log_level", type=int, default=3, choices=range(1, 11), metavar="[1-10]",
                        help="Optimizer parameter. Only effects SMAC and GPGO. Sets how often log messages appear. "
                             "Lower values mean more messages.")
    parser.add_argument('--path_baselines', default='pyoptes/optimization/budget_allocation/blackbox_learning/baselines/',
                        help="")
    parser.add_argument('--path_networks', default='../../networks/data',
                        help='Location where the networks are saved to. '
                             'Path on cluster. /p/projects/ou/labs/gane/optes/mcmc_100nets/data'
                             '/p/projects/ou/labs/gane/optes/mcmc_100nets/data/')

    args = parser.parse_args()

    # prepare the directory for the plots, hyperparameters and results
    if not os.path.exists(args.path_baselines):
        os.makedirs(args.path_baselines)


    #
    nodes = [120, 1040, 57590]
    # needs a dynamic list
    sentinels = []

    statistics = [mean_tia, rms_tia, percentile_tia]

    budget_scale = [1, 4, 12]
    for n_nodes in nodes:
        if n_nodes == 57590:
            n_runs = 10
        else:
            n_runs = 100
        for scale in budget_scale:
            # The total budget is a function of the number of nodes in the network and
            total_budget = args.scale_total_budget * args.n_nodes  # i.e., on average, nodes will do one test per year
            for statistic in statistics:


                pass

    # lists for result aggregations
    list_prior = []

    list_best_otf = []  # best optimizer function value on each network and corresponding standard error
    list_best_otf_stderr = []
    list_baseline_otf = []  # baseline  function value on each network and corresponding standard error
    list_baseline_otf_stderr = []

    list_ratio_otf = []     # ratio of best optimizer function value to baseline function value on each network
    list_best_solution_history = []
    list_stderr_history = []

    list_all_prior_tf = []
    list_all_prior_stderr = []

    list_time_for_optimization = []

    for n in range(n_runs):

        # load networks and setup the target function
        transmissions, capacities, degrees = create_graph(n, args.graph_type, args.n_nodes, args.path_networks)

        f.prepare(n_nodes=args.n_nodes,
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
                                       sentinels=args.sentinels,
                                       mixed_strategies=False,
                                       only_baseline=False)

        # evaluate the strategies in the prior
        list_prior_tf = []
        list_prior_stderr = []

        for i, p in tqdm(enumerate(prior), leave=False, total=len(prior)):

            p = map_low_dim_x_to_high_dim(x=p,
                                          number_of_nodes=args.n_nodes,
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


