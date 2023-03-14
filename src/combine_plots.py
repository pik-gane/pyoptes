import numpy as np
import os
import json
import pylab as plt
from pyoptes import bo_load_raw_data, bo_compute_average_otf_and_stderr


def plotting(data, labels, name_plot, path):
    plt.clf()
    colors = ['orange', 'blue', 'green', 'red']

    print(f'\n{name_plot}')

    for i, d in enumerate(data):
        path_experiment = os.path.join(path, d)

        raw_data_path = os.path.join(path_experiment, 'raw_data/')

        individual_runs = os.listdir(raw_data_path)

        experiment_params = os.path.join(path_experiment, "experiment_hyperparameters.json")
        with open(experiment_params, 'r') as f:
            hyperparameters = json.load(f)

        optimizer = hyperparameters['optimizer_hyperparameters']['optimizer']
        network_type = hyperparameters['simulation_hyperparameters']['graph']
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

        baseline_mean, baseline_stderr = bo_compute_average_otf_and_stderr(list_otf=list_baseline_otf,
                                                                           list_stderr=list_baseline_otf_stderr,
                                                                           n_runs=n_runs)

        optimizer_history, stderr_history = bo_compute_average_otf_and_stderr(
            list_otf=list_best_solution_history,
            list_stderr=list_stderr_history,
            n_runs=n_runs)

        b = np.ones(len(optimizer_history)) * baseline_mean
        ratio_history = 100-(100*(optimizer_history / b))
        print(list_best_otf, np.shape(list_best_otf))
        print([np.min(g) for g in list_best_solution_history])
        print(optimizer_history, np.shape(optimizer_history))
        assert list_best_otf == [np.min(g) for g in list_best_solution_history]
        print(np.mean(list_best_otf), np.mean([np.min(g) for g in list_best_solution_history]))
        # print(ratio_history)
        # print('optimizer',optimizer_history)
        # print('baseline',b)
        # print('min pos',[np.argmin(g) for g in list_best_solution_history])
        # print('min ',[np.min(g) for g in list_best_solution_history], np.mean([np.min(g) for g in list_best_solution_history]))
        # print(list_baseline_otf)
        print('--')

        plt.plot(range(len(optimizer_history)), ratio_history, label=labels[i], color=colors[i])

        plt.title(f'{name_plot}') #TODO add useful information to the title
        plt.xlabel('Iteration')
        plt.ylabel('r_o') #TODO more useful label

    plt.plot(range(len(b)), np.ones(len(b)), label='baseline', color='black')
    plt.legend()
    plt.savefig(os.path.join(path, 'combined_plots', f'{name_plot}.png'),
                dpi=300)
    plt.clf()


if __name__ == '__main__':

    path = '../data/blackbox_learning/results/'

    # gpgo 1040 default + 4N budget. 12N budget + UCB
    # Load the data
    data = ['20230226_gpgo_mean_nodes_1040',
            # '20230226_gpgo_mean_nodes_1040_budget_4N', # missing
            # '20230226_gpgo_mean_nodes_1040_budget_12N', # missing
            # '20230226_gpgo_mean_nodes_1040_UCB']
            ]
    labels = ['default',
              'budget 4N', 'budget 12N', 'UCB']

    name_plot = 'gpgo_1040'

    plotting(data, labels, name_plot, path)

    # gpgo 57590 normal + 4N. 12N budget + x

    data = ['20230226_gpgo_mean_nodes_57590_sentinels_1329',
            '20230226_gpgo_mean_nodes_57590_sentinels_1329_budget_4N',
            '20230226_gpgo_mean_nodes_57590_sentinels_1329_budget_12N',]

    labels = ['default', 'budget 4N', 'budget 12N']

    name_plot = 'gpgo_57590'

    # plotting(data, labels, name_plot, path)

    # cma 1040 normal + 4N. 12N budget + x

    data = ['20230109_cma_mean_nodes_1040',
            # '20230226_cma_mean_nodes_1040_budget_4N',
            # '20230226_cma_mean_nodes_1040_budget_12N']
            ]
    labels = ['default',
              'budget 4N', 'budget 12N']

    name_plot = 'cma_1040'

    plotting(data, labels, name_plot, path)

    print(dsfsadf)
    # cma 57590 normal + 4N. 12N budget + x

    data = ['20230226_cma_mean_nodes_57590_sentinels_1329',
            '20230301_cma_mean_nodes_57590_sentinels_1329_budget_4N',
            '20230301_cma_mean_nodes_57590_sentinels_1329_budget_12N']

    labels = ['default', 'budget 4N', 'budget 12N']

    name_plot = 'cma_57590'

    plotting(data, labels, name_plot, path)

    # np 1040 normal + 4N. 12N budget + x

    data = ['20230120_np_mean_nodes_1040',
            '20230206_np_mean_nodes_1040_4N_budget',
            '20230206_np_mean_nodes_1040_12N_budget']

    labels = ['default', 'budget 4N', 'budget 12N']

    name_plot = 'np_1040'

    plotting(data, labels, name_plot, path)

    # np 57590 normal + 4N. 12N budget + x

    data = ['20230301_np_mean_nodes_57590_sentinels_1329',
            '20230301_np_mean_nodes_57590_4N_budget',
            '20230301_np_mean_nodes_57590_12N_budget']

    labels = ['default', 'budget 4N', 'budget 12N']

    name_plot = 'np_57590'

    plotting(data, labels, name_plot, path)

    # combined 1040 and 57590

    data = [#'20230226_gpgo_mean_nodes_1040',
            '20230109_cma_mean_nodes_1040',
            '20230309_np_mean_nodes_1040_50_iterations'] #TODO add NP data

    labels = [#'gpgo',
              'cma', 'np']

    name_plot = 'combined_1040'

    plotting(data, labels, name_plot, path)

    data = ['20230226_gpgo_mean_nodes_57590_sentinels_1329', #TODO fix length of gpgo, just pelicate the data
            '20230226_cma_mean_nodes_57590_sentinels_1329',
            '20230308_np_mean_nodes_57590_sentinels_1329_50_iterations']

    labels = ['gpgo', 'cma', 'np']

    name_plot = 'combined_57590'

    plotting(data, labels, name_plot, path)






