'''
Create plots containing the results of multiple optimizers
'''

import numpy as np
import os
import pylab as plt
import argparse
import glob
import json
from tqdm import tqdm
from pyoptes import load_raw_data, compute_average_otf_and_stderr
from pyoptes import plot_optimizer_history_with_two_baselines, plot_prior, plot_multiple_optimizer

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # TODO add a argument for the name of the plotS

    parser.add_argument('--path_plot', default='pyoptes/optimization/budget_allocation/blackbox_learning/plots/',
                        help="Optimizer parameter. Location where all the individual results"
                             " of the optimizers are saved to. "
                             "Default location is 'pyoptes/optimization/budget_allocation/blackbox_learning/plots/'")
    # TODO not quite sure how to use this yet, maybe take a list as input?
    # parser.add_argument("--name_experiment",
    #                     help="The name of the folder where the results of the optimizer run are saved to.")

    parser.add_argument("--sentinels", type=int, default=1040,
                        help="Set the number of nodes that are used. Has to be smaller than or equal to n_nodes. "
                             "Default is 120 nodes.")
    parser.add_argument("--n_nodes", type=int, default=1040, choices=[120, 1040, 57590],
                        help="Si-simulation parameter. "
                             "Defines the number of nodes used by the SI-model to create a graph. "
                             "Default value is 120 nodes.")

    # TODO keep in mind max_iterations may change between experiments
    # in case these want to be compared the shorter experiment can be extended to the same length by repeating the last value
    parser.add_argument("--max_iterations", type=int, default=50,
                        help="Optimizer parameter. The maximum number of iterations the algorithms run.")

    parser.add_argument('--acquisition_function', default='EI',
                        choices=['EI', 'PI', 'UCB', 'Entropy', 'tEI'],
                        help='GPGO optimizer parameter. Defines the acquisition function that is used by GPGO.')
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

    parser.add_argument('--popsize', type=int, default=9,
                        help='CMA-ES optimizer parameter. Defines the size of the population each iteration.'
                             'CMA default is "4+int(3*log(n_nodes))" '
                             '-> 18 of 120, 24 for 1040, 36 for 57590.'
                             'Is set to 9 for performance reasons.')
    parser.add_argument('--scale_sigma', type=float, default=0.25,
                        help='CMA-ES optimizer parameter. Defines the scaling of the standard deviation. '
                             'Default is a standard deviation of 0.25 of the total budget.')
    parser.add_argument('--cma_prior', type=int, default=0,
                        help='CMA-ES optimizer parameter. Sets which test strategy in the prior is used as the initial '
                             'population for cma.')

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

    parser.add_argument('--mode_choose_sentinels', choices=['degree', 'capacity', 'transmission'], default='degree',
                        help='Sets the mode of how sentinels are chosen. ')
    parser.add_argument("--log_level", type=int, default=3, choices=range(1, 11), metavar="[1-10]",
                        help="Optimizer parameter. Only effects SMAC and GPGO. Sets how often log messages appear. "
                             "Lower values mean more messages.")

    args = parser.parse_args()

    # TODO make this work for multiple optimizers
    # TODO maybe set desired parameters with command line arguments, then search the plots directory for all
    # all experiments satisfying the desired settings

    paths_experiment_params = glob.glob(os.path.join(args.path_plot, '**/experiment_hyperparameters.json'))

    data_optimizer = []

    # TODO glob all directories, but use if statments to extract only data specified in arguments (optimizer, statistic, etc.)
    # or give the name of directories directly
    for experiment_params in tqdm(paths_experiment_params):

        with open(experiment_params, 'r') as f:
            hyperparameters = json.load(f)

        # compare the set of desired parameters with the ones in the experiment_params file

        arguments_dict = vars(args)

        optimizer = hyperparameters['optimizer_hyperparameters']['optimizer']
        network_type = hyperparameters['simulation_hyperparameters']['graph']
        n_runs = hyperparameters['simulation_hyperparameters']['n_runs']
        n_nodes = hyperparameters['simulation_hyperparameters']['n_nodes']
        sentinels = hyperparameters['simulation_hyperparameters']['sentinels']

        statistic = hyperparameters['simulation_hyperparameters']['statistic']

    # # save date of the experiment for plotting
    #     raw_data = load_raw_data(os.path.join(path_data, 'raw_data/'))
    #
    #     # compute the averages of the c_raw_data
    #     optimizer_history, stderr_history = compute_average_otf_and_stderr(raw_data['list_best_solution_history'],
    #                                                                        raw_data['list_stderr_history'],
    #                                                                        n_runs)
    #
    #     baseline_mean, baseline_stderr = compute_average_otf_and_stderr(raw_data['list_baseline_otf'],
    #                                                                     raw_data['list_baseline_otf_stderr'],
    #                                                                     n_runs)
    #
    #     prior_mean, prior_stderr = compute_average_otf_and_stderr(raw_data['list_all_prior_tf'],
    #                                                               raw_data['list_all_prior_stderr'],
    #                                                               n_runs)
    #
    #     do = {'optimizer_history': optimizer_history,
    #           'stderr_history': stderr_history,
    #           'optimizer': optimizer}
    #
    #     data_optimizer.append(do)
    #
    # # as the baseline are taken from the last experiment in the loop, it is not advised to mix experiments with different
    # # number of nodes
    # # TODO add option to plot multiple baselines
    # data_baseline = [{'baseline_mean': baseline_mean,
    #                   'baseline_stderr': baseline_stderr,
    #                   'name': 'uniform'},
    #                  {'baseline_mean': prior_mean[1],
    #                   'baseline_stderr': prior_stderr[1],
    #                   'name': 'highest degree'}]
    #
    # plot_multiple_optimizer(args.path_plot, data_optimizer, data_baseline, n_nodes, sentinels)


