'''
Create plots containing the results of multiple optimizers
'''

import numpy as np
import os
import pylab as plt
import glob
import json
from tqdm import tqdm
from pyoptes import bo_load_raw_data, bo_compute_average_otf_and_stderr
from pyoptes import bo_plot_optimizer_history_with_two_baselines, bo_plot_prior, bo_plot_multiple_optimizer


def bbo_combined_plots(path_plot,
                       optimizer,
                       n_nodes,
                       sentinels,
                       max_iterations,
                       acquisition_function,
                       use_prior,
                       prior_only_baseline,
                       prior_mixed_strategies,
                       popsize,
                       scale_sigma,
                       statistic,
                       n_simulations,
                       graph_type,
                       scale_total_budget,
                       mode_choose_sentinels):

    # TODO make this work for multiple optimizers
    # TODO maybe set desired parameters with command line arguments, then search the plots directory for all
    # all experiments satisfying the desired settings

    paths_experiment_params = glob.glob(os.path.join(path_plot, '**/experiment_hyperparameters.json'), recursive=True)

    data_optimizer = []

    # TODO glob all directories, but use if statments to extract only data specified in arguments (optimizer, statistic, etc.)
    # or give the name of directories directly
    for experiment_params in tqdm(paths_experiment_params):

        with open(experiment_params, 'r') as f:
            hyperparameters = json.load(f)

        # compare the set of desired parameters with the ones in the experiment_params file

        # arguments_dict = vars(args)

        optimizer = hyperparameters['optimizer_hyperparameters']['optimizer']
        max_iterations = hyperparameters['optimizer_hyperparameters']['max_iterations']

        network_type = hyperparameters['simulation_hyperparameters']['graph']
        n_runs = hyperparameters['simulation_hyperparameters']['n_runs']
        n_nodes = hyperparameters['simulation_hyperparameters']['n_nodes']
        sentinels = hyperparameters['simulation_hyperparameters']['sentinels']
        statistic = hyperparameters['simulation_hyperparameters']['statistic']
        total_budget = hyperparameters['simulation_hyperparameters']['total_budget']

        param_check = statistic == statistic and n_nodes == n_nodes \
                      and sentinels == sentinels and max_iterations == max_iterations \
                      and total_budget == scale_total_budget*n_nodes

        if optimizer == optimizer and param_check \
                or optimizer == 'all' and param_check:
            # get the path to the experiment
            path_experiment = os.path.split(experiment_params)[0]
            # save date of the experiment for plotting
            raw_data = bo_load_raw_data(os.path.join(path_experiment, 'raw_data/'))

            # compute the averages of the c_raw_data
            optimizer_history, stderr_history = bo_compute_average_otf_and_stderr(raw_data['list_best_solution_history'],
                                                                                  raw_data['list_stderr_history'],
                                                                                  n_runs)

            baseline_mean, baseline_stderr = bo_compute_average_otf_and_stderr(raw_data['list_baseline_otf'],
                                                                               raw_data['list_baseline_otf_stderr'],
                                                                               n_runs)

            prior_mean, prior_stderr = bo_compute_average_otf_and_stderr(raw_data['list_all_prior_tf'],
                                                                         raw_data['list_all_prior_stderr'],
                                                                         n_runs)

            do = {'optimizer_history': optimizer_history,
                  'stderr_history': stderr_history,
                  'optimizer': optimizer}

            data_optimizer.append(do)
        else:
            # TODO missing errorhandling

            print('s')
    #
    print(np.shape(data_optimizer))
    # as the baseline are taken from the last experiment in the loop, it is not advised to mix experiments with different
    # number of nodes
    # TODO add option to plot multiple baselines
    data_baseline = [{'baseline_mean': baseline_mean,
                      'baseline_stderr': baseline_stderr,
                      'name': 'uniform'},
                     {'baseline_mean': prior_mean[1],
                      'baseline_stderr': prior_stderr[1],
                      'name': 'highest degree'}]

    bo_plot_multiple_optimizer(path_plot, data_optimizer, data_baseline, n_nodes, sentinels)


