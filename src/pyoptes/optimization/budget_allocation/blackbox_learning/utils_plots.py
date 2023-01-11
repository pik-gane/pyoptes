import os
import numpy as np
import pylab as plt


def bo_plot_time_for_optimization(time_for_optimization: list, path_experiment: str, optimizer: str,
                                  file_name='time_for_optimization.png',
                                  title='Time for objective function evaluation of',
                                  sum_up_time=False):
    """

    @param sum_up_time:
    @param title:
    @param file_name:
    @param time_for_optimization:
    @param path_experiment:
    @param optimizer:
    """
    if sum_up_time:
        time_for_optimization = np.cumsum(time_for_optimization, axis=0)
    plt.clf()
    plt.plot(range(len(time_for_optimization)), time_for_optimization)
    plt.title(title+f' {optimizer}')
    plt.xlabel('Optimizer iteration')
    plt.ylabel('Cumulative time in minutes')

    plt.savefig(os.path.join(path_experiment, file_name))
    plt.clf()


def bo_plot_optimizer_history(optimizer_history: list, stderr_history: list,
                              baseline_mean: float, baseline_stderr: float,
                              n_nodes: int,
                              sentinels: int,
                              path_experiment: str,
                              optimizer: str,
                              name='_plot') -> None:
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
             linestyle='dotted', color='black')
    plt.plot(range(len(optimizer_history)), s_bounds[:, 1],
             linestyle='dotted', color='black')

    b = np.ones(len(optimizer_history)) * baseline_mean
    plt.plot(range(len(optimizer_history)), b, label='uniform baseline')
    # add standard error of the baseline
    plt.plot(range(len(optimizer_history)), b + baseline_stderr,
             linestyle='dotted', color='black')
    plt.plot(range(len(optimizer_history)), b - baseline_stderr,
             linestyle='dotted', color='black')

    plt.title(f'{optimizer}, {n_nodes} nodes, {sentinels} sentinels')
    plt.xlabel('Iteration')
    plt.ylabel('Number of infected animals')
    plt.legend()

    plt.savefig(os.path.join(path_experiment, f'{optimizer}{name}.png'))
    plt.clf()


def bo_plot_optimizer_history_with_two_baselines(optimizer_history: list, stderr_history: list,
                                                 baseline_mean: list, baseline_stderr: list,
                                                 prior_mean: list, prior_stderr: list,
                                                 n_nodes: int,
                                                 sentinels: int,
                                                 path_experiment: str,
                                                 optimizer: str,
                                                 name: str = '_plot') -> None:
    """

    @param optimizer_history:
    @param stderr_history:
    @param baseline_mean:
    @param baseline_stderr:
    @param prior_mean:
    @param prior_stderr:
    @param n_nodes:
    @param sentinels:
    @param path_experiment:
    @param optimizer:
    @param name:
    """
    plt.clf()
    # plot the trajectory of the optimizer
    stderr_bounds = np.array([[m + s, m - s] for m, s in zip(optimizer_history, stderr_history)])
    plt.plot(range(len(optimizer_history)), optimizer_history, label=optimizer)
    # add standard error of the mean
    plt.plot(range(len(optimizer_history)), stderr_bounds[:, 0],
             linestyle='dotted', color='black')
    plt.plot(range(len(optimizer_history)), stderr_bounds[:, 1],
             linestyle='dotted', color='black')

    # plot the uniform baseline
    b = np.ones(len(optimizer_history)) * baseline_mean
    plt.plot(range(len(optimizer_history)), b, label='uniform baseline')
    # add standard error of the baseline
    plt.plot(range(len(optimizer_history)), b + baseline_stderr,
             linestyle='dotted', color='black')
    plt.plot(range(len(optimizer_history)), b - baseline_stderr,
             linestyle='dotted', color='black')

    # plot the highest degree baseline
    b = np.ones(len(optimizer_history)) * prior_mean[0]
    plt.plot(range(len(optimizer_history)), b, label='uniform sentinels baseline')
    # add standard error of the highest degree baseline
    plt.plot(range(len(optimizer_history)), b + prior_stderr[0],
             linestyle='dotted', color='black')
    plt.plot(range(len(optimizer_history)), b - prior_stderr[0],
             linestyle='dotted', color='black')

    # plot the highest degree baseline
    b = np.ones(len(optimizer_history)) * prior_mean[1]
    plt.plot(range(len(optimizer_history)), b, label='highest degree baseline s/6')
    # add standard error of the highest degree baseline
    plt.plot(range(len(optimizer_history)), b + prior_stderr[1],
             linestyle='dotted', color='black')
    plt.plot(range(len(optimizer_history)), b - prior_stderr[1],
             linestyle='dotted', color='black')

    # # plot the highest capacity baseline
    # b = np.ones(len(optimizer_history)) * prior_mean[2]
    # plt.plot(range(len(optimizer_history)), b, label='highest capacity baseline s/6')
    # # add standard error of the highest degree baseline
    # plt.plot(range(len(optimizer_history)), b + prior_stderr[2],
    #          linestyle='dotted', color='black')
    # plt.plot(range(len(optimizer_history)), b - prior_stderr[2],
    #          linestyle='dotted', color='black')

    # plot the highest degree baseline
    b = np.ones(len(optimizer_history)) * prior_mean[3]
    plt.plot(range(len(optimizer_history)), b, label='highest degree baseline s/12')
    # add standard error of the highest degree baseline
    plt.plot(range(len(optimizer_history)), b + prior_stderr[3],
             linestyle='dotted', color='black')
    plt.plot(range(len(optimizer_history)), b - prior_stderr[3],
             linestyle='dotted', color='black')

    # # plot the highest capacity baseline
    # b = np.ones(len(optimizer_history)) * prior_mean[4]
    # plt.plot(range(len(optimizer_history)), b, label='highest capacity baseline s/12')
    # # add standard error of the highest degree baseline
    # plt.plot(range(len(optimizer_history)), b + prior_stderr[4],
    #          linestyle='dotted', color='black')
    # plt.plot(range(len(optimizer_history)), b - prior_stderr[4],
    #          linestyle='dotted', color='black')
    #
    # plot the highest degree baseline
    b = np.ones(len(optimizer_history)) * prior_mean[5]
    plt.plot(range(len(optimizer_history)), b, label='highest degree baseline s/24')
    # add standard error of the highest degree baseline
    plt.plot(range(len(optimizer_history)), b + prior_stderr[5],
             linestyle='dotted', color='black')
    plt.plot(range(len(optimizer_history)), b - prior_stderr[5],
             linestyle='dotted', color='black')
    #
    # # plot the highest capacity baseline
    # b = np.ones(len(optimizer_history)) * prior_mean[6]
    # plt.plot(range(len(optimizer_history)), b, label='highest capacity baseline s/24')
    # # add standard error of the highest degree baseline
    # plt.plot(range(len(optimizer_history)), b + prior_stderr[5],
    #          linestyle='dotted', color='black')
    # plt.plot(range(len(optimizer_history)), b - prior_stderr[5],
    #          linestyle='dotted', color='black')

    plt.title(f'{optimizer}, {n_nodes} nodes, {sentinels} sentinels')
    plt.xlabel('Iteration')
    plt.ylabel('Number of infected animals')
    plt.legend()

    plot_path = os.path.join(path_experiment, f'{optimizer}{name}.png')
    plt.savefig(plot_path)
    plt.clf()


def bo_plot_prior(path_experiment: str,
                  n_nodes: int,
                  y_prior_mean: np.array, y_prior_stderr: np.array,
                  n_runs: int) -> None:
    """

    @param n_runs:
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


def bo_plot_effect_of_different_sentinels(number_of_sentinels: list,
                                          m: list, stderr: list,
                                          n_nodes: int,
                                          path_experiment: str,
                                          minimum: list,
                                          title: str = '',
                                          mode_choose_sentinels: str = 'degree',
                                          step_size: int = 5
                                          ) -> None:
    """

    @param step_size:
    @param minimum:
    @param mode_choose_sentinels:
    @param number_of_sentinels:
    @param m:
    @param stderr:
    @param n_nodes:
    @param path_experiment:
    @param title:
    """
    # create the labeling for the x-axis by hand to include the number of sentinels were the objective function is minimal
    new_step_size = int(n_nodes / (step_size * 6) * step_size)
    x_axis_ticks = list(range(0, n_nodes+new_step_size, new_step_size))
    x_axis_ticks.append(minimum[0])
    x_axis_ticks = sorted(x_axis_ticks)

    plt.clf()
    plt.xticks(x_axis_ticks)
    plt.axvspan(minimum[0]-int(step_size/2),
                minimum[0]+int(step_size/2), alpha=0.5, color='red', label='minimum')
    plt.bar(number_of_sentinels, m)
    plt.errorbar(number_of_sentinels, m, yerr=stderr, fmt='o', color="r")
    plt.title(title)
    plt.xlabel('Number of sentinels')
    plt.ylabel('Infected animals')
    plt.legend()

    plt.savefig(os.path.join(path_experiment, f'Sentinel_budgets_{n_nodes}_nodes_mode_{mode_choose_sentinels}.png'))
    plt.clf()


def bo_plot_multiple_optimizer(path_experiment: str,
                               data_optimizer: list,
                               data_baseline: list,
                               n_nodes: int,
                               sentinels: int) -> None:
    """
    Plot any number of optimizers against any number baselines
    @param sentinels:
    @param n_nodes:
    @param path_experiment:
    @param data_optimizer:
    @param data_baseline:
    """

    plt.clf()
    for d in data_optimizer:
        optimizer_history = d['optimizer_history']
        stderr_history = d['stderr_history']
        optimizer = d['optimizer']

        # plot the trajectory of the optimizer
        stderr_bounds = np.array([[m + s, m - s] for m, s in zip(optimizer_history, stderr_history)])
        # TODO label has to be changed. If gpgo occurs twice the labels are useless
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

    plt.title(f'Optimizer performance against baseline, {n_nodes} nodes, {sentinels} sentinels')
    plt.xlabel('Iteration')
    plt.ylabel('Number of infected animals')
    plt.legend()

    plot_path = os.path.join(path_experiment, f'combined_optimizer_{n_nodes}_n_nodes.png')
    plt.savefig(plot_path)
    plt.clf()


def bo_scatter_plot(path_experiment: str,
                    data_x: list, data_y: list,
                    plot_name: str,
                    plot_title: str,
                    x_label: str,
                    y_label: str):
    """

    @param path_experiment:
    @param data_x:
    @param data_y:
    @param plot_name:
    @param plot_title:
    @param x_label:
    @param y_label:
    """
    plt.clf()
    plt.scatter(data_x, data_y)
    plt.title(plot_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(os.path.join(path_experiment, plot_name))
    plt.clf()
