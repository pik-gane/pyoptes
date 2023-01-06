'''
Searches for all experiments and documents them in a csv file.
'''

import os
import glob
import json
import csv
from tqdm import tqdm
import numpy as np


def bbo_document_experiments(path_plot):

    # grabs also experiments in test folders. They should be deleted before running this
    # also grabs incomplete experiments
    path_to_experiments = glob.glob(os.path.join(path_plot, '**/experiment_hyperparameters.json'), recursive=True)

    param_set = set()

    with open('../data/blackbox_learning/experiments.csv', 'w', newline='') as csvfile:
        cswriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        # get all unique hyperparameter for all experiments
        for experiment_params in path_to_experiments:

            with open(experiment_params, 'r') as f:
                hyperparameters = json.load(f)

            optimizer_hyperparameters = hyperparameters['optimizer_hyperparameters']
            simulation_hyperparameters = hyperparameters['simulation_hyperparameters']

            optim_param_set = set(optimizer_hyperparameters.keys())
            sim_param_set = set(simulation_hyperparameters.keys())

            param_set = param_set.union(optim_param_set)
            param_set = param_set.union(sim_param_set)

        # write header, these are the hyperparameters that are shared by every experiment
        # and will be the first columns to make the csv more readable
        start_of_row = ['optimizer', 'n_nodes', 'sentinels', 'max_iterations', 'total_budget', 'statistic',
                        'ratio', 'otf', 'baseline', 'time (in hours)']
        # remove start_of_row from param_set
        param_set = param_set.difference(set(start_of_row))
        param_list = list(param_set)

        # extend start_of_row with param_list to create new param_list with all parameters in the desired order
        param_list = start_of_row + sorted(param_list)

        cswriter.writerow(param_list)

        # write entries for each experiment
        for experiment_params in tqdm(path_to_experiments):

            path_to_eval_output = os.path.join(os.path.dirname(experiment_params), 'evaluation_output.txt')
            if os.path.isfile(path_to_eval_output):

                # read the optimization results from the text-file
                with open(path_to_eval_output, 'r') as f:
                    lines = f.readlines()
                ratio_otf_baseline = np.round(float(lines[1].split(':')[-1]), decimals=2)
                baseline = np.round(float(lines[2].split(':')[-1].split(',')[0]), decimals=2)
                otf = np.round(float(lines[3].split(':')[-1].split(',')[0]), decimals=2)
                time_to_optimize = np.round(float(lines[4].split(':')[-1]), decimals=2)
                evaluation_results = {'ratio': ratio_otf_baseline,
                                      'baseline': baseline,
                                      'otf': otf,
                                      'time (in hours)': time_to_optimize}

                # read the hyperparameter for the .json-file
                with open(experiment_params, 'r') as f:
                    hyperparameters = json.load(f)

                optimizer_hyperparameters = hyperparameters['optimizer_hyperparameters']
                simulation_hyperparameters = hyperparameters['simulation_hyperparameters']

                row = []
                for param in param_list:
                    if param in optimizer_hyperparameters:
                        row.append(optimizer_hyperparameters[param])
                    elif param in simulation_hyperparameters:
                        row.append(simulation_hyperparameters[param])
                    elif param in evaluation_results:
                        row.append(evaluation_results[param])
                    else:
                        row.append('')

                cswriter.writerow(row)
            else:
                pass
