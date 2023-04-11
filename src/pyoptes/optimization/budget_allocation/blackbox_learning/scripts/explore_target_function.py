import json
from pyoptes import bo_plot_effect_of_different_sentinels
import os


def load_data(path_data):
    explo_data = []

    for pd in path_data:
        with open(os.path.join(path, pd), 'r') as fp:
            data = json.load(fp)
        # convert the string in the loaded json to the types above
        data['sentinels'] = list(map(int, data['sentinels']))
        data['mean'] = list(map(float, data['mean']))
        data['stderr'] = list(map(float, data['stderr']))
        data['minimum'] = [int(list(data['minimum'])[0]), float(list(data['minimum'])[1])]
        data['n_nodes'] = int(data['n_nodes'])
        data['n_runs'] = list(map(int, data['n_runs']))
        data['n_simulations'] = int(data['n_simulations'])
        data['scale_total_budget'] = int(data['scale_total_budget'])
        data['step_size'] = int(data['step_size'])
        data['statistic'] = str(data['statistic'])
        data['mode_choose_sentinels'] = str(data['mode_choose_sentinels'])

        explo_data.append(data)

    return explo_data


if __name__=="__main__":

    path = '../data/blackbox_learning/results/exploration'

    path_data = ['120_nodes_syn_graph_capacity_mode.json',
                 '120_nodes_syn_graph_degree_mode.json',
                 '120_nodes_ba_graph_degree_mode.json',
                 '120_nodes_waxman_graph_degree_mode.json']

    explo_data = load_data(path_data)

    bo_plot_effect_of_different_sentinels(data=explo_data,
                                          path_experiment=path,
                                          title=f'Simulations with increasing number of sentinels. 120 nodes')
    # ---------------------------------------------------------------------------

    path_data = ['1040_nodes_syn_graph_capacity_mode.json',
                 '1040_nodes_syn_graph_degree_mode.json',
                 '1040_nodes_ba_graph_degree_mode.json',
                 '1040_nodes_waxman_graph_degree_mode.json']

    explo_data = load_data(path_data)

    bo_plot_effect_of_different_sentinels(data=explo_data,
                                          path_experiment=path,
                                          title=f'Simulations with increasing number of sentinels. 1040 nodes')

    # ---------------------------------------------------------------------------

    path_data = ['57590_nodes_syn_graph_capacity_mode.json',
                    '57590_nodes_syn_graph_degree_mode.json',]

    explo_data = load_data(path_data)

    bo_plot_effect_of_different_sentinels(data=explo_data,
                                            path_experiment=path,
                                            title=f'Simulations with increasing number of sentinels. 57590 nodes')
