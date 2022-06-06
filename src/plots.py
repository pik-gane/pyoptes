import numpy as np
import os
import pylab as plt
import argparse
import glob
import json
from pyoptes import load_raw_data, compute_average_otf_and_stderr

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("name_experiment",
                        help="The name of the folder where the results of the optimizer run are saved to.")

    parser.add_argument('--path_plot', default='pyoptes/optimization/budget_allocation/blackbox_learning/plots/',
                        help="Optimizer parameter. Location where all the individual results"
                             " of the optimizers are saved to. "
                             "Default location is 'pyoptes/optimization/budget_allocation/blackbox_learning/plots/'")

    args = parser.parse_args()

    base_path = os.path.join(args.path_plot, args.name_experiment)

    experiment_params = os.path.join(base_path, 'experiment_hyperparameters.json')
    with open(experiment_params, 'r') as f:
        hyperparameters = json.load(f)

    optimizer = hyperparameters['optimizer_hyperparameters']['optimizer']
    network_type = hyperparameters['simulation_hyperparameters']['graph']
    n_runs = hyperparameters['simulation_hyperparameters']['n_runs']
    n_nodes = hyperparameters['simulation_hyperparameters']['n_nodes']
    sentinels = hyperparameters['simulation_hyperparameters']['sentinels']

    path_raw_data = os.path.join(base_path, 'raw_data/')
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

    print(prior_mean)
    print(baseline_mean)
    print(prior_stderr)
    print(baseline_stderr)

    print(np.shape(optimizer_history), np.shape(stderr_history))

    s_bounds = np.array([[m+s, m-s] for m, s in zip(optimizer_history, stderr_history)])

    plt.clf()
    plt.plot(range(len(optimizer_history)), optimizer_history, label=optimizer)
    # add standard error of the mean
    plt.plot(range(len(optimizer_history)), s_bounds[:, 0],
             linestyle='dotted', color='black', label=f'stderr {optimizer}')
    plt.plot(range(len(optimizer_history)), s_bounds[:, 1],
             linestyle='dotted', color='black')

    b = np.ones(len(optimizer_history)) * baseline_mean
    plt.plot(range(len(optimizer_history)), b, label='baseline')
    # add standard error of the baseline
    plt.plot(range(len(optimizer_history)), b + baseline_stderr,
             label='stderr baseline', linestyle='dotted', color='red')
    plt.plot(range(len(optimizer_history)), b - baseline_stderr,
             linestyle='dotted', color='red')

    plt.title(f'{optimizer}, {n_nodes} nodes, {sentinels} sentinels')
    plt.xlabel('Iteration')
    plt.ylabel('Number of infected animals') # TODO change name to something more descriptive, like number of infected animals
    plt.legend()
    plt.show()