import json
import os
import glob
from pyoptes import bo_plot_effect_of_different_sentinels


def bbo_plot_target_function(path_plot: str = '../data/blackbox_learning/results/'):

    # load json files from /data/blackbox_learning/exploration
    path = os.path.join(path_plot, 'exploration/')
    files = glob.glob(os.path.join(path, '*.json'))

    data120 = []
    data1040 = []
    data57590 = []

    # load json files from files
    for f in files:
        with open(f, 'r') as fp:
            data = json.load(fp)

            if data['n_nodes'] == 120:
                data120.append(data)
            elif data['n_nodes'] == 1040:
                data1040.append(data)
            elif data['n_nodes'] == 57590:
                data57590.append(data)
            else:
                raise ValueError(f'n_nodes {data["n_nodes"]} not supported')

    bo_plot_effect_of_different_sentinels(data=data120,
                                          title=f'Simulations with increasing number of sentinels. {data120[0]["n_nodes"]} nodes',
                                          path_experiment=path)

    bo_plot_effect_of_different_sentinels(data=data1040,
                                          title=f'Simulations with increasing number of sentinels. {data1040[0]["n_nodes"]} nodes',
                                          path_experiment=path)

    bo_plot_effect_of_different_sentinels(data=data57590,
                                          title=f'Simulations with increasing number of sentinels. {data57590[0]["n_nodes"]} nodes',
                                          path_experiment=path)
