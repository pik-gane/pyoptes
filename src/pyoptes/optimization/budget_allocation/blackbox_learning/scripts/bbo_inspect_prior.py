'''
Computes the objective function value of the prior for each strategy and the standard error of the mean for each strategy.
'''
import numpy as np
import glob
import pylab as plt
import os
from pyoptes import bo_compute_average_otf_and_stderr, bo_load_raw_data
from tqdm import tqdm
import json


# TODO can probably be moved into bo_postprocessing
def bbo_inspect_prior(path_plot):

    paths_experiment_params = glob.glob(os.path.join(path_plot, '**/experiment_hyperparameters.json'), recursive=True)
    for experiment_params in tqdm(paths_experiment_params):

        # get experiment specific hyperparameters
        with open(experiment_params, 'r') as f:
            hyperparameters = json.load(f)

        n_runs = hyperparameters['simulation_hyperparameters']['n_runs']

        experiment_directory = os.path.split(experiment_params)[0]

        raw_data_path = os.path.join(experiment_directory, 'raw_data/')

        individual_runs = os.listdir(raw_data_path)

        list_best_otf = []  # best optimizer function value on each network and corresponding standard error
        list_best_otf_stderr = []
        list_baseline_otf = []  # baseline  function value on each network and corresponding standard error
        list_baseline_otf_stderr = []

        list_ratio_otf = []  # ratio of best optimizer function value to baseline function value on each network
        list_best_solution_history = []
        list_stderr_history = []

        list_all_prior_tf = []
        list_all_prior_stderr = []

        # combine the raw data from the (separate) experiments into one list each
        for r in individual_runs:

            p = os.path.join(raw_data_path, r)
            raw_data = bo_load_raw_data(p)

            # extend the list in raw_data to the appropriate list in the outer loop
            list_baseline_otf.extend(raw_data['list_baseline_otf'])
            list_baseline_otf_stderr.extend(raw_data['list_baseline_otf_stderr'])

            list_ratio_otf.extend(raw_data['list_ratio_otf'])

            list_best_solution_history.extend(raw_data['list_best_solution_history'])
            list_stderr_history.extend(raw_data['list_stderr_history'])

            list_best_otf.extend(raw_data['list_best_otf'])
            list_best_otf_stderr.extend(raw_data['list_best_otf_stderr'])

            list_all_prior_tf.extend(raw_data['list_all_prior_tf'])
            list_all_prior_stderr.extend(raw_data['list_all_prior_stderr'])

        av_prior_tf, av_prior_stderr = bo_compute_average_otf_and_stderr(list_otf=list_all_prior_tf,
                                                                         list_stderr=list_all_prior_stderr,
                                                                         n_runs=len(list_all_prior_tf))

        baseline_mean, baseline_stderr = bo_compute_average_otf_and_stderr(list_otf=list_baseline_otf,
                                                                           list_stderr=list_baseline_otf_stderr,
                                                                           n_runs=n_runs)

        b = np.ones(len(av_prior_tf)) * baseline_mean
        plt.plot(range(len(b)), b, label='baseline', color='black')
        plt.bar(range(len(av_prior_tf)), av_prior_tf, label='prior')
        plt.errorbar(range(len(av_prior_tf)), av_prior_tf, yerr=av_prior_stderr, fmt='o', color="r")
        plt.title(f'Objective function evaluation for {len(av_prior_tf)} strategies,\naverage over {len(list_all_prior_tf)} networks')
        plt.xlabel('Prior')
        plt.ylabel('Number of infected animals')
        plt.savefig(os.path.join(experiment_directory, 'objective_function_values_prior.png'),
                    dpi=300)
        plt.clf()