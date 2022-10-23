import numpy as np
import os
from pyoptes import load_raw_data, compute_average_otf_and_stderr, save_results
from pyoptes import plot_optimizer_history, plot_time_for_optimization
import json

if __name__ == '__main__':

    # TODO temp substitute for all dirs in results
    for i in range(2):
        print('\n', i)

        path_experiment = '../data/blackbox_learning/results/test/'

        raw_data_path = os.path.join(path_experiment, 'raw_data/')

        runs = os.listdir(raw_data_path)

        # check whether the experiment has been run successfully
        experiment_complete = True
        for r in runs:  # TODO better name needed
            p = os.path.join(raw_data_path, r, '.done')
            if not os.path.isfile(p):
                experiment_complete = False
        if not experiment_complete:
            print(f'Experiment "{path_experiment}" incomplete.')
            break

        experiment_params = os.path.join(path_experiment, "experiment_hyperparameters.json")
        with open(experiment_params, 'r') as f:
            hyperparameters = json.load(f)

        optimizer = hyperparameters['optimizer_hyperparameters']['optimizer']
        network_type = hyperparameters['simulation_hyperparameters']['graph']
        n_runs = hyperparameters['simulation_hyperparameters']['n_runs']
        n_nodes = hyperparameters['simulation_hyperparameters']['n_nodes']
        sentinels = hyperparameters['simulation_hyperparameters']['sentinels']

        list_prior = []

        list_best_otf = []  # best optimizer function value on each network and corresponding standard error
        list_best_otf_stderr = []
        list_baseline_otf = []  # baseline  function value on each network and corresponding standard error
        list_baseline_otf_stderr = []

        list_ratio_otf = []  # ratio of best optimizer function value to baseline function value on each network
        list_best_solution_history = []
        list_stderr_history = []

        list_all_prior_tf = []
        list_all_prior_stderr = []

        list_time_for_optimization = []
        list_time_acquisition_optimization = []
        list_time_update_surrogate = []

        # load the raw data from the experiment and compute the average OTF and STDERR
        for r in runs:

            p = os.path.join(raw_data_path, r)
            raw_data = load_raw_data(p)

            # extend the list in raw_data to the appropriate list in the outer loop
            list_best_otf.extend(raw_data['list_best_otf'])
            list_best_otf_stderr.extend(raw_data['list_best_otf_stderr'])
            list_baseline_otf.extend(raw_data['list_baseline_otf'])
            list_baseline_otf_stderr.extend(raw_data['list_baseline_otf_stderr'])

            list_ratio_otf.extend(raw_data['list_ratio_otf'])
            list_best_solution_history.extend(raw_data['list_best_solution_history'])
            list_stderr_history.extend(raw_data['list_stderr_history'])

            list_all_prior_tf.extend(raw_data['list_all_prior_tf'])
            list_all_prior_stderr.extend(raw_data['list_all_prior_stderr'])

            list_time_for_optimization.extend(raw_data['list_time_for_optimization'])
            list_time_acquisition_optimization.extend(raw_data['list_time_acquisition_optimization'])
            list_time_update_surrogate.extend(raw_data['list_time_update_surrogate'])

        # ------------------------------------------------------------
        # postprocessing of all runs
        # ------------------------------------------------------------
        # compute the averages over all optimization runs of the prior, the optimizer, the baseline and their standard error
        average_prior_tf, average_prior_stderr = compute_average_otf_and_stderr(list_otf=list_all_prior_tf,
                                                                                list_stderr=list_all_prior_stderr,
                                                                                n_runs=n_runs)
        # compute the average OTFs, baseline and their standard errors
        average_best_otf, average_best_otf_stderr = compute_average_otf_and_stderr(list_otf=list_best_otf,
                                                                                   list_stderr=list_best_otf_stderr,
                                                                                   n_runs=n_runs)

        average_baseline, average_baseline_stderr = compute_average_otf_and_stderr(list_otf=list_baseline_otf,
                                                                                   list_stderr=list_baseline_otf_stderr,
                                                                                   n_runs=n_runs)

        average_ratio_otf = np.mean(list_ratio_otf)

        # create an average otf plot
        average_best_solution_history, average_stderr_history = compute_average_otf_and_stderr(
            list_otf=list_best_solution_history,
            list_stderr=list_stderr_history,
            n_runs=n_runs)

        plot_optimizer_history(optimizer_history=average_best_solution_history,
                               stderr_history=average_stderr_history,
                               baseline_mean=average_baseline, baseline_stderr=average_baseline_stderr,
                               n_nodes=n_nodes, sentinels=sentinels,
                               path_experiment=path_experiment, optimizer=optimizer,
                               name='_average_plot')

        time_for_optimization = np.mean(list_time_for_optimization, axis=0)
        time_for_optimization = np.cumsum(time_for_optimization, axis=0)
        plot_time_for_optimization(time_for_optimization=time_for_optimization,
                                   path_experiment=path_experiment, optimizer=optimizer,
                                   file_name='time_for_optimization.png',
                                   title='Time for optimization',
                                   sum_up_time=False)

        output = f'Results averaged over {n_runs} optimizer runs' \
                 f'\naverage ratio otf to baseline: {average_ratio_otf}' \
                 f'\naverage baseline and stderr: {average_baseline}, {average_baseline_stderr}' \
                 f'\naverage best strategy OTF and stderr: {average_best_otf}, {average_best_otf_stderr}' \
                 f'\nTime for optimization (in hours): {time_for_optimization[-1] / 60}'

        save_results(best_test_strategy=None,
                     save_test_strategy=False,
                     path_experiment=path_experiment,
                     output=output)
        print(output)

        # ---- misc ----
        if optimizer == 'gpgo' or optimizer == 'np':
            # plot the average time the optimization of the acquisition function takes
            # as well as the update of the surrogate function
            time_acquisition_optimization = np.mean(list_time_acquisition_optimization, axis=0)
            time_update_surrogate = np.mean(list_time_update_surrogate, axis=0)

            plot_time_for_optimization(time_for_optimization=time_acquisition_optimization,
                                       path_experiment=path_experiment, optimizer=optimizer,
                                       file_name='time_for_acquisition_optimization.png',
                                       title='Average time for acquisition function optimization',
                                       sum_up_time=True)
            plot_time_for_optimization(time_for_optimization=time_update_surrogate,
                                       path_experiment=path_experiment, optimizer=optimizer,
                                       file_name='time_for_surrogate_update.png',
                                       title='Average time for surrogate function update',
                                       sum_up_time=True)
