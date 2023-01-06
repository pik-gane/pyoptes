'''
The postprocessing script for the blackbox optimization. Has to run after the optimization has finished.
Creates plots and saves them to the experiment directory.

'''

import numpy as np
import os
from pyoptes import bo_load_raw_data, bo_compute_average_otf_and_stderr, bo_save_results
from pyoptes import bo_plot_optimizer_history, bo_plot_time_for_optimization
import json
import glob


def bbo_postprocessing(path_plot, force_postprocessing=False):

    # get all experiments in the folder with a json file
    path_to_experiments = glob.glob(os.path.join(path_plot, '**/experiment_hyperparameters.json'), recursive=True)

    # iterate over all experiments
    for experiment_params in path_to_experiments:

        # check the main path, whether the experiment has been processed already
        # by checking for the existence of the evaluation_output.txt file
        path_evaluation_output = os.path.join(os.path.dirname(experiment_params), 'evaluation_output.txt')
        if os.path.exists(path_evaluation_output) or not force_postprocessing:
            print('Experiment already processed')
        else:

            path_experiment = os.path.dirname(experiment_params)

            raw_data_path = os.path.join(path_experiment, 'raw_data/')

            individual_runs = os.listdir(raw_data_path)

            # experiment_params = os.path.join(path_experiment, "experiment_hyperparameters.json")
            with open(experiment_params, 'r') as f:
                hyperparameters = json.load(f)

            optimizer = hyperparameters['optimizer_hyperparameters']['optimizer']
            # network_type = hyperparameters['simulation_hyperparameters']['graph']
            n_runs = hyperparameters['simulation_hyperparameters']['n_runs']
            n_nodes = hyperparameters['simulation_hyperparameters']['n_nodes']
            sentinels = hyperparameters['simulation_hyperparameters']['sentinels']

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

            # combine the raw data from the (separate) experiments into one list each
            for r in individual_runs:

                p = os.path.join(raw_data_path, r)
                raw_data = bo_load_raw_data(p)

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
            # compute the averages over all optimization runs of the prior,
            # the optimizer, the baseline and their standard error
            # TODO prior stuff is not needed anymore
            # average_prior_tf, average_prior_stderr = compute_average_otf_and_stderr(list_otf=list_all_prior_tf,
            #                                                                         list_stderr=list_all_prior_stderr,
            #                                                                         n_runs=n_runs)
            # compute the average OTFs, baseline and their standard errors
            average_best_otf, average_best_otf_stderr = bo_compute_average_otf_and_stderr(list_otf=list_best_otf,
                                                                                          list_stderr=list_best_otf_stderr,
                                                                                          n_runs=n_runs)

            average_baseline, average_baseline_stderr = bo_compute_average_otf_and_stderr(list_otf=list_baseline_otf,
                                                                                          list_stderr=list_baseline_otf_stderr,
                                                                                          n_runs=n_runs)

            average_ratio_otf = np.mean(list_ratio_otf)

            # create an average otf plot
            average_best_solution_history, average_stderr_history = bo_compute_average_otf_and_stderr(
                list_otf=list_best_solution_history,
                list_stderr=list_stderr_history,
                n_runs=n_runs)

            bo_plot_optimizer_history(optimizer_history=average_best_solution_history,
                                      stderr_history=average_stderr_history,
                                      baseline_mean=average_baseline, baseline_stderr=average_baseline_stderr,
                                      n_nodes=n_nodes, sentinels=sentinels,
                                      path_experiment=path_experiment, optimizer=optimizer,
                                      name='_average_plot')

            time_for_optimization = np.mean(list_time_for_optimization, axis=0)
            time_for_optimization = np.cumsum(time_for_optimization, axis=0)
            bo_plot_time_for_optimization(time_for_optimization=time_for_optimization,
                                          path_experiment=path_experiment, optimizer=optimizer,
                                          file_name='time_for_optimization.png',
                                          title='Time for optimization',
                                          sum_up_time=False)

            output = f'Results averaged over {n_runs} optimizer runs' \
                     f'\naverage ratio otf to baseline: {average_ratio_otf}' \
                     f'\naverage baseline and stderr: {average_baseline}, {average_baseline_stderr}' \
                     f'\naverage best strategy OTF and stderr: {average_best_otf}, {average_best_otf_stderr}' \
                     f'\nTime for optimization (in hours): {time_for_optimization[-1] / 60}'

            bo_save_results(best_test_strategy=None,
                            save_test_strategy=False,
                            path_experiment=path_experiment,
                            output=output)
            print('--> finished postprocessing')
            print(output, '\n')

            # ---- misc ----
            if optimizer == 'gpgo' or optimizer == 'np':
                # plot the average time the optimization of the acquisition function takes
                # as well as the update of the surrogate function
                time_acquisition_optimization = np.mean(list_time_acquisition_optimization, axis=0)
                time_update_surrogate = np.mean(list_time_update_surrogate, axis=0)

                bo_plot_time_for_optimization(time_for_optimization=time_acquisition_optimization,
                                              path_experiment=path_experiment, optimizer=optimizer,
                                              file_name='time_for_acquisition_optimization.png',
                                              title='Average time for acquisition function optimization',
                                              sum_up_time=True)
                bo_plot_time_for_optimization(time_for_optimization=time_update_surrogate,
                                              path_experiment=path_experiment, optimizer=optimizer,
                                              file_name='time_for_surrogate_update.png',
                                              title='Average time for surrogate function update',
                                              sum_up_time=True)
