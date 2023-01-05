import numpy as np
import os
import glob
import json
from tqdm import tqdm
from pyoptes import bo_load_raw_data, bo_compute_average_otf_and_stderr
from pyoptes import bo_plot_optimizer_history_with_two_baselines, bo_plot_prior, bo_plot_multiple_optimizer, bo_plot_time_for_optimization


def bbo_create_individual_plots(path_plot):

    paths_experiment_params = glob.glob(os.path.join(path_plot, '**/experiment_hyperparameters.json'))
    for experiment_params in tqdm(paths_experiment_params):
        # get experiment specific hyperparameters
        with open(experiment_params, 'r') as f:
            hyperparameters = json.load(f)

        optimizer = hyperparameters['optimizer_hyperparameters']['optimizer']
        network_type = hyperparameters['simulation_hyperparameters']['graph']
        n_runs = hyperparameters['simulation_hyperparameters']['n_runs']
        n_nodes = hyperparameters['simulation_hyperparameters']['n_nodes']
        sentinels = hyperparameters['simulation_hyperparameters']['sentinels']

        statistic = hyperparameters['simulation_hyperparameters']['statistic']

        # get the path to the experiment
        path_experiment = os.path.split(experiment_params)[0]
        experiment_directory = os.path.split(experiment_params)[0]
        experiment_name = os.path.split(experiment_directory)[1][9:]

        #
        add_to_plots = True
        if optimizer == 'gpgo':
            if hyperparameters['optimizer_hyperparameters']['prior_only_baseline'] == True:
                add_to_plots = False
            else:
                add_to_plots = True
        else:
            add_to_plots = True

        if add_to_plots:

            # load the raw data from the experiment and compute the average OTF and STDERR
            # for the optimizer, the baseline and the prior
            path_raw_data = os.path.join(path_experiment, 'raw_data/')
            raw_data = bo_load_raw_data(path_raw_data)

            optimizer_history, stderr_history = bo_compute_average_otf_and_stderr(raw_data['list_best_solution_history'],
                                                                                  raw_data['list_stderr_history'],
                                                                                  n_runs)

            baseline_mean, baseline_stderr = bo_compute_average_otf_and_stderr(raw_data['list_baseline_otf'],
                                                                               raw_data['list_baseline_otf_stderr'],
                                                                               n_runs)

            prior_mean, prior_stderr = bo_compute_average_otf_and_stderr(raw_data['list_all_prior_tf'],
                                                                         raw_data['list_all_prior_stderr'],
                                                                         n_runs)

            # ---------------------------------------------------------------------------------------------------------
            # plot optimizer history against uniform and highest degree baseline
            plot_name = '_average_plot2'
            bo_plot_optimizer_history_with_two_baselines(optimizer_history, stderr_history,
                                                         baseline_mean, baseline_stderr,
                                                         prior_mean, prior_stderr,
                                                         n_nodes, sentinels, path_experiment, optimizer, plot_name)

            # create a bar plot of all strategies in the prior
            bo_plot_prior(path_experiment, n_nodes, prior_mean, prior_stderr, n_runs)

            try:
                print('shape raw data list time acquisition optimization', np.shape(raw_data['list_time_acquisition_optimization']))
                print('shape raw data list time update surrogate', np.shape(raw_data['list_time_update_surrogate']))
                bo_plot_time_for_optimization(raw_data['list_time_acquisition_optimization'][0],
                                              path_experiment,
                                              optimizer,
                                              file_name='time_for_acquisition_optimization.png',
                                              sum_up_time=True)

                bo_plot_time_for_optimization(raw_data['list_time_update_surrogate'][0],
                                              path_experiment,
                                              optimizer,
                                              file_name='time_for_surrogate_update.png',
                                              sum_up_time=True)
            except:
                print('no time data available')

            # # ---------------------------------------------------------------------------------------------------------
            # # plot multiple optimizers
            # path_plot = os.path.join(path_experiment, 'plots/')
            # if not os.path.exists(path_plot):
            #     os.makedirs(path_plot)
            #
            # # create a plot with all the optimizers
            # plot_multiple_optimizer(path_plot, n_nodes, n_runs, optimizer, network_type, statistic, sentinels)

            # TODO plot prior for each optimizer

            # TODO add the 'default' plot fro each experiment