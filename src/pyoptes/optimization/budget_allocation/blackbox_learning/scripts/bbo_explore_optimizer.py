'''
Visualize how the optimizers move through the search space with each iteration.

'''

from pyoptes import create_graph, compute_average_otf_and_stderr

import glob
import json
import os
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import euclidean
import pylab as plt


def bbo_explore_optimizer(path_plot):

    path_to_optimizer = '../../pyoptes_plots/*/experiment_hyperparameters.json'

    # glob optimizer hyperparameters
    optimizer_hyperparameters = glob.glob(path_to_optimizer)

    for experiment_params in optimizer_hyperparameters:

        # get experiment specific hyperparameters
        with open(experiment_params, 'r') as f:
            hyperparameters = json.load(f)

        optimizer = hyperparameters['optimizer_hyperparameters']['optimizer']
        network_type = hyperparameters['simulation_hyperparameters']['graph']
        n_runs = hyperparameters['simulation_hyperparameters']['n_runs']
        n_nodes = hyperparameters['simulation_hyperparameters']['n_nodes']
        sentinels = hyperparameters['simulation_hyperparameters']['sentinels']
        total_budget = hyperparameters['simulation_hyperparameters']['total_budget']

        experiment_directory = os.path.split(experiment_params)[0]
        experiment_name = os.path.split(experiment_directory)[1][9:]
        print(experiment_name)
        uniform_baseline = np.array([total_budget / n_nodes for _ in range(n_nodes)])

        # iterate over n_runs and load the data
        for n in tqdm(range(n_runs)):
            p = os.path.join(experiment_directory, 'individual', f'{n}', 'raw_data', 'test_strategies_history')
            p = glob.glob(os.path.join(p,  f'*.npy'))
            # p contains budgets the optimizer used in each iteration
            for strategy in p:
                budgets = np.load(strategy)

                dists = [euclidean(b, uniform_baseline) for b in budgets]
                # print min, max, mean pof dists
                print(np.min(dists), np.max(dists), np.mean(dists), np.median(dists))

                plt.clf()
                plt.scatter(list(range(len(budgets))), dists)
                plt.show()
                plt.clf()
            break