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

    c = os.path.join(args.path_plot, '20220611_cma_rms_nodes_1040')
    g = os.path.join(args.path_plot, '20220611_gpgo_rms_nodes_1040')

    n_runs = 100
    c_raw_data = load_raw_data(os.path.join(c, 'raw_data/'))

    # compute the averages of the c_raw_data
    optimizer_history, stderr_history = compute_average_otf_and_stderr(c_raw_data['list_best_solution_history'],
                                                                       c_raw_data['list_stderr_history'],
                                                                       n_runs)

    baseline_mean, baseline_stderr = compute_average_otf_and_stderr(c_raw_data['list_baseline_otf'],
                                                                    c_raw_data['list_baseline_otf_stderr'],
                                                                    n_runs)

    prior_mean, prior_stderr = compute_average_otf_and_stderr(c_raw_data['list_all_prior_tf'],
                                                              c_raw_data['list_all_prior_stderr'],
                                                              n_runs)

    g_raw_data = load_raw_data(os.path.join(g, 'raw_data/'))

    # compute the averages of the g_raw_data
    optimizer_history_g, stderr_history_g = compute_average_otf_and_stderr(g_raw_data['list_best_solution_history'],
                                                                          g_raw_data['list_stderr_history'],
                                                                            n_runs)

    baseline_mean_g, baseline_stderr_g = compute_average_otf_and_stderr(g_raw_data['list_baseline_otf'],
                                                                          g_raw_data['list_baseline_otf_stderr'],
                                                                            n_runs)

    prior_mean_g, prior_stderr_g = compute_average_otf_and_stderr(g_raw_data['list_all_prior_tf'],
                                                                    g_raw_data['list_all_prior_stderr'],
                                                                    n_runs)

    # print np.shape of optimizer, baseline and prior (mean and stderr) for c and g
    print('c:')
    print('optimizer_history:', np.shape(optimizer_history))
    print('baseline_mean:', np.shape(baseline_mean))
    print('prior_mean:', np.shape(prior_mean))
    print('stderr_history:', np.shape(stderr_history))
    print('baseline_stderr:', np.shape(baseline_stderr))
    print('prior_stderr:', np.shape(prior_stderr))
    print('g:')
    print('optimizer_history_g:', np.shape(optimizer_history_g))
    print('baseline_mean_g:', np.shape(baseline_mean_g))
    print('prior_mean_g:', np.shape(prior_mean_g))
    print('stderr_history_g:', np.shape(stderr_history_g))
    print('baseline_stderr_g:', np.shape(baseline_stderr_g))
    print('prior_stderr_g:', np.shape(prior_stderr_g))

    # print baseline mean and stderr for c and g
    print('c:')
    print('baseline_mean:', baseline_mean)
    print('baseline_stderr:', baseline_stderr)
    print('g:')
    print('baseline_mean_g:', baseline_mean_g)
    print('baseline_stderr_g:', baseline_stderr_g)

    # print prior mean and stderr for c and g
    print('c:')
    print('prior_mean:', prior_mean)
    print('prior_stderr:', prior_stderr)
    print('g:')
    print('prior_mean_g:', prior_mean_g)
    print('prior_stderr_g:', prior_stderr_g)

    data_optimizer = [{'optimizer_history': optimizer_history,
                       'stderr_history': stderr_history,
                       'optimizer': 'cma',},
                      {'optimizer_history': optimizer_history_g,
                       'stderr_history': stderr_history_g,
                       'optimizer': 'gpgo',}]

    data_baseline = [{'baseline_mean': baseline_mean,
                      'baseline_stderr': baseline_stderr,
                      'name': 'uniform',},
                     {'baseline_mean': prior_mean_g[1],
                      'baseline_stderr': prior_stderr_g[1],
                      'name': 'highest degree',}]

    plt.clf()
    for d in data_optimizer:
        optimizer_history = d['optimizer_history']
        stderr_history = d['stderr_history']
        optimizer = d['optimizer']

        # plot the trajectory of the optimizer
        stderr_bounds = np.array([[m + s, m - s] for m, s in zip(optimizer_history, stderr_history)])
        plt.plot(range(len(optimizer_history)), optimizer_history, label=optimizer)
        # add standard error of the mean
        plt.plot(range(len(optimizer_history)), stderr_bounds[:, 0],
                 linestyle='dotted', color='black')
        plt.plot(range(len(optimizer_history)), stderr_bounds[:, 1],
                 linestyle='dotted', color='black')

    for d in data_baseline:
        baseline_mean = d['baseline_mean']
        baseline_stderr = d['baseline_stderr']
        name = d['name']

        # plot the uniform baseline
        b = np.ones(len(optimizer_history)) * baseline_mean
        plt.plot(range(len(optimizer_history)), b, label=name)
        # add standard error of the baseline
        plt.plot(range(len(optimizer_history)), b + baseline_stderr,
                 linestyle='dotted', color='black')
        plt.plot(range(len(optimizer_history)), b - baseline_stderr,
                 linestyle='dotted', color='black')

    plt.title(f'd')
    plt.xlabel('Iteration')
    plt.ylabel('Number of infected animals')
    plt.legend()
    plt.show()
    # paths_experiment_params = glob.glob(os.path.join(args.path_plot, '**/experiment_hyperparameters.json'))
    # for experiment_params in tqdm(paths_experiment_params):
    #     # get experiment specific hyperparameters
    #     with open(experiment_params, 'r') as f:
    #         hyperparameters = json.load(f)
    #
    #     optimizer = hyperparameters['optimizer_hyperparameters']['optimizer']
    #     network_type = hyperparameters['simulation_hyperparameters']['graph']
    #     n_runs = hyperparameters['simulation_hyperparameters']['n_runs']
    #     n_nodes = hyperparameters['simulation_hyperparameters']['n_nodes']
    #     sentinels = hyperparameters['simulation_hyperparameters']['sentinels']
    #
    #     statistic = hyperparameters['simulation_hyperparameters']['statistic']
    #
    #     # get the path to the experiment
    #     path_experiment = os.path.split(experiment_params)[0]
    #
    #     # load the raw data from the experiment and compute the average OTF and STDERR
    #     # for the optimizer, the baseline and the prior
    #     path_raw_data = os.path.join(path_experiment, 'raw_data/')
    #     raw_data = load_raw_data(path_raw_data)
    #
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
        # # plot optimizer history against uniform and highest degree baseline
        # plot_name = '_average_plot2'
        # plot_optimizer_history_with_two_baselines(optimizer_history, stderr_history,
        #                                           baseline_mean, baseline_stderr,
        #                                           prior_mean, prior_stderr,
        #                                           n_nodes, sentinels, path_experiment, optimizer, plot_name)
        #
        # # create a bar plot of all strategies in the prior
        # plot_prior(path_experiment, n_nodes, prior_mean, prior_stderr, n_runs)

