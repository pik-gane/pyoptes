import os
import numpy as np
import pylab as plt
from time import time


def plot_time_for_optimization(time_for_optimization, n_nodes, sentinels, path_experiment, optimizer):
    """

    @param time_for_optimization:
    @param n_nodes:
    @param sentinels:
    @param path_experiment:
    @param optimizer:
    """
    plt.clf()
    plt.plot(range(len(time_for_optimization)), time_for_optimization)
    plt.title(f'Time for objective function evaluation with {optimizer}, {n_nodes} nodes, {sentinels} sentinels')
    plt.xlabel('Iteration')
    plt.ylabel('Time in minutes')
    plt.savefig(os.path.join(path_experiment, 'time_for_optimization.png'))
    plt.clf()


def plot_optimizer_history(optimizer_history, stderr_history, baseline_mean, baseline_stderr,
                           n_nodes, sentinels, path_experiment, optimizer, name='_plot'):
    """

    @param name:
    @param optimizer_history:
    @param stderr_history:
    @param baseline_mean:
    @param baseline_stderr:
    @param n_nodes:
    @param sentinels:
    @param path_experiment:
    @param optimizer:
    """

    s_bounds = np.array([[m+s, m-s] for m, s in zip(optimizer_history, stderr_history)])

    plt.clf()
    plt.plot(range(len(optimizer_history)), optimizer_history, label=optimizer)
    # add standard error of the mean
    plt.plot(range(len(optimizer_history)), s_bounds[:, 0],
             linestyle='dotted', color='black', label=f'stderr {optimizer}')
    plt.plot(range(len(optimizer_history)), s_bounds[:, 1],
             linestyle='dotted', color='black')

    b = np.ones(len(optimizer_history)) * baseline_mean
    plt.plot(range(len(optimizer_history)), b, label='uniform baseline')
    # add standard error of the baseline
    plt.plot(range(len(optimizer_history)), b + baseline_stderr,
             label='stderr uniform baseline', linestyle='dotted', color='red')
    plt.plot(range(len(optimizer_history)), b - baseline_stderr,
             linestyle='dotted', color='red')

    plt.title(f'{optimizer}, {n_nodes} nodes, {sentinels} sentinels')
    plt.xlabel('Iteration')
    plt.ylabel('Number of infected animals') # TODO change name to something more descriptive, like number of infected animals
    plt.legend()
    plt.savefig(os.path.join(path_experiment, f'{optimizer}{name}.png'))
    plt.clf()


def plot_optimizer_history_with_two_baselines(optimizer_history, stderr_history, baseline_mean, baseline_stderr,
                                              prior_mean, prior_stderr, n_nodes, sentinels,
                                              path_experiment, optimizer, name='_plot'):

    plot_path = os.path.join(path_experiment, f'{optimizer}{name}.png')

    plt.clf()
    # plot the trajectory of the optimizer
    stderr_bounds = np.array([[m + s, m - s] for m, s in zip(optimizer_history, stderr_history)])
    plt.plot(range(len(optimizer_history)), optimizer_history, label=optimizer)
    # add standard error of the mean
    plt.plot(range(len(optimizer_history)), stderr_bounds[:, 0],
             linestyle='dotted', color='black', label=f'stderr {optimizer}')
    plt.plot(range(len(optimizer_history)), stderr_bounds[:, 1],
             linestyle='dotted', color='black')

    # plot the uniform baseline
    b = np.ones(len(optimizer_history)) * baseline_mean
    plt.plot(range(len(optimizer_history)), b, label='uniform baseline')
    # add standard error of the baseline
    plt.plot(range(len(optimizer_history)), b + baseline_stderr,
             label='stderr uniform baseline', linestyle='dotted', color='red')
    plt.plot(range(len(optimizer_history)), b - baseline_stderr,
             linestyle='dotted', color='red')

    # plot the highest degree baseline
    b = np.ones(len(optimizer_history)) * prior_mean[1]
    plt.plot(range(len(optimizer_history)), b, label='highest degree baseline')
    # add standard error of the highest degree baseline
    plt.plot(range(len(optimizer_history)), b + prior_stderr[1],
             label='stderr highest degree baseline', linestyle='dotted', color='green')
    plt.plot(range(len(optimizer_history)), b - prior_stderr[1],
             linestyle='dotted', color='green')

    plt.title(f'{optimizer}, {n_nodes} nodes, {sentinels} sentinels')
    plt.xlabel('Iteration')
    plt.ylabel('Number of infected animals')
    plt.legend()
    plt.savefig(plot_path)


def plot_prior(path_experiment, n_nodes, y_prior_mean, y_prior_stderr, n_runs):
    """

    @param y_prior_stderr:
    @param n_nodes:
    @param y_prior_mean:
    @param path_experiment:

    """

    min_y_prior_mean = y_prior_mean.min()
    max_y_prior_mean = y_prior_mean.max()
    plt.clf()

    plt.bar(range(len(y_prior_mean)), y_prior_mean, label='prior')
    plt.errorbar(range(len(y_prior_mean)), y_prior_mean, yerr=y_prior_stderr, fmt='o', color="r")
    plt.title(f'Objective function evaluation for {len(y_prior_mean)} strategies, average over {n_runs} networks')
    plt.xlabel('Prior')
    plt.ylabel('Infected animals')
    # TODO move text in the top right corner of the plot
    plt.text(25, 1000, f'min: {min_y_prior_mean:2f}\nmax: {max_y_prior_mean:2f}',
             bbox=dict(facecolor='red', alpha=0.5))
    plt.savefig(os.path.join(path_experiment, f'objective_function_values_prior_{n_nodes}_nodes.png'))
    plt.clf()
