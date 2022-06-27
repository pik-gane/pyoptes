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

    # TODO make this work for multiple optimizers

    p = glob.glob(os.path.join(args.path_plot, '**nodes_1040*'))

    print(p)
    print(daf)

    c = os.path.join(args.path_plot, '20220611_cma_rms_nodes_1040')
    g = os.path.join(args.path_plot, '20220611_gpgo_rms_nodes_1040')

    path_data_optimizer = [c, g]

    data_optimizer = []
    data_baseline = []

    # TODO glob all directories, but use if statments to extract only data specified in arguments (optimizer, statistic, etc.)
    # or give the name of directories directly
    for path_data in path_data_optimizer:

        experiment_params = os.path.join(path_data, 'experiment_hyperparameters.json')
        with open(experiment_params, 'r') as f:
            hyperparameters = json.load(f)

        optimizer = hyperparameters['optimizer_hyperparameters']['optimizer']
        network_type = hyperparameters['simulation_hyperparameters']['graph']
        n_runs = hyperparameters['simulation_hyperparameters']['n_runs']
        n_nodes = hyperparameters['simulation_hyperparameters']['n_nodes']
        sentinels = hyperparameters['simulation_hyperparameters']['sentinels']

        statistic = hyperparameters['simulation_hyperparameters']['statistic']

        raw_data = load_raw_data(os.path.join(path_data, 'raw_data/'))

        # compute the averages of the c_raw_data
        optimizer_history, stderr_history = compute_average_otf_and_stderr(raw_data['list_best_solution_history'],
                                                                           raw_data['list_stderr_history'],
                                                                           n_runs)

        baseline_mean, baseline_stderr = compute_average_otf_and_stderr(raw_data['list_baseline_otf'],
                                                                        raw_data['list_baseline_otf_stderr'],
                                                                        n_runs)

        prior_mean, prior_stderr = compute_average_otf_and_stderr(raw_data['list_all_prior_tf'],
                                                                  raw_data['list_all_prior_stderr'],
                                                                  n_runs)

        do = {'optimizer_history': optimizer_history,
              'stderr_history': stderr_history,
              'optimizer': optimizer}

        data_optimizer.append(do)

    # TODO add option to plot multiple baselines
    data_baseline = [{'baseline_mean': baseline_mean,
                      'baseline_stderr': baseline_stderr,
                      'name': 'uniform'},
                     {'baseline_mean': prior_mean[1],
                      'baseline_stderr': prior_stderr[1],
                      'name': 'highest degree'}]

    plot_multiple_optimizer(args.path_plot, data_optimizer, data_baseline, n_nodes, sentinels)


