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

    parser.add_argument('--path_plot', default='pyoptes/optimization/budget_allocation/blackbox_learning/plots/',
                        help="Optimizer parameter. Location where all the individual results"
                             " of the optimizers are saved to. "
                             "Default location is 'pyoptes/optimization/budget_allocation/blackbox_learning/plots/'")

    args = parser.parse_args()

    paths_experiment_params = glob.glob(os.path.join(args.path_plot, '**/experiment_hyperparameters.json'))
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

        # load the raw data from the experiment and compute the average OTF and STDERR
        # for the optimizer, the baseline and the prior
        path_raw_data = os.path.join(path_experiment, 'raw_data/')
        raw_data = load_raw_data(path_raw_data)

        optimizer_history, stderr_history = compute_average_otf_and_stderr(raw_data['list_best_solution_history'],
                                                                           raw_data['list_stderr_history'],
                                                                           n_runs)

        baseline_mean, baseline_stderr = compute_average_otf_and_stderr(raw_data['list_baseline_otf'],
                                                                        raw_data['list_baseline_otf_stderr'],
                                                                        n_runs)

        prior_mean, prior_stderr = compute_average_otf_and_stderr(raw_data['list_all_prior_tf'],
                                                                  raw_data['list_all_prior_stderr'],
                                                                  n_runs)

        # ---------------------------------------------------------------------------------------------------------
        # plot optimizer history against uniform and highest degree baseline
        plot_name = '_average_plot2'
        plot_optimizer_history_with_two_baselines(optimizer_history, stderr_history,
                                                  baseline_mean, baseline_stderr,
                                                  prior_mean, prior_stderr,
                                                  n_nodes, sentinels, path_experiment, optimizer, plot_name)

        # create a bar plot of all strategies in the prior
        plot_prior(path_experiment, n_nodes, prior_mean, prior_stderr, n_runs)

        # TODO plot prior for each optimizer

        # TODO add the 'default' plot fro each experiment
