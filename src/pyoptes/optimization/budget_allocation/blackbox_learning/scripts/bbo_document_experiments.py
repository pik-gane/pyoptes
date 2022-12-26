'''
Searches for all experiments and documents them in a csv file.
'''

import argparse
import os
import glob
import json
import csv


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

        # TODO header needs to be extended with rows for optimizer performance metrics, e.g baseline, otf values, ratio
        # write header
        start_of_row = ['optimizer', 'n_nodes', 'sentinels', 'max_iterations', 'total_budget', 'statistic']
        # remove start_of_row from param_set
        param_set = param_set.difference(set(start_of_row))
        param_list = list(param_set)

        # extend start_of_row with param_list to create new param_list
        param_list = start_of_row + sorted(param_list)

        cswriter.writerow(param_list)

        # write entries for each experiment
        for experiment_params in path_to_experiments:
            # TODO add check for 'evaluation_output.txt' file, if not present, skip experiment
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
                else:
                    row.append('')

            cswriter.writerow(row)
