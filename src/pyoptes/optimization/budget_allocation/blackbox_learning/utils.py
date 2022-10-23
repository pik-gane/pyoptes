import os
import json
import numpy as np
from tqdm import tqdm
import pandas as pd
import networkx as nx
from scipy.stats.mstats import mjci


def softmax(x: np.array) -> np.array:
    """
    Softmax function.
    @param x: budget vector, numpy array
    @return: scaled budget vector, numpy array
    """
    x = x - np.max(x)  # for numerical stability. Prevents infinities in the denominator.
    x = np.exp(x) / sum(np.exp(x))
    return x


def compute_average_otf_and_stderr(list_otf: list,
                                   list_stderr: list,
                                   n_runs: int) -> (float, float):
    average_best_otf = np.mean(list_otf, axis=0)
    s = np.mean(list_stderr, axis=0)
    v = np.var(list_otf, axis=0)
    average_best_otf_stderr = np.sqrt(v/n_runs + s**2)

    return average_best_otf, average_best_otf_stderr


def save_results(best_test_strategy: np.array,
                 path_experiment: str,
                 output: str, save_test_strategy: bool = True) -> None:
    """
    Saves the evaluation of the output and the corresponding best parameter.
    @param save_test_strategy:
    @param output:
    @param best_test_strategy: numpy array, the best parameter the optimizer has found
    @param path_experiment:
    """
    if not os.path.isdir(path_experiment):
        os.makedirs(path_experiment)

    with open(os.path.join(path_experiment, 'evaluation_output.txt'), 'w') as text_file:
        text_file.write(output)

    if save_test_strategy:
        np.save(os.path.join(path_experiment, 'best_parameter'), best_test_strategy)


def save_hyperparameters(hyperparameters: dict,
                         base_path: str,) -> None:
    """
    Saves the parameters for a training run of a model,
    Parameters are saved as a json file.
    Args:
        hyperparameters: dict,
        base_path: string, base path where the results are saved to
    """
    if not os.path.isdir(base_path):
        os.makedirs(base_path)

    json_path = os.path.join(base_path, 'experiment_hyperparameters.json')

    with open(json_path, 'w') as fp:
        json.dump(hyperparameters, fp, sort_keys=True, indent=4)
    print('saved hyperparameters\n')


# TODO add function to load raw data from file
def save_raw_data(list_best_otf: list, list_best_otf_stderr: list,
                  list_baseline_otf: list, list_baseline_otf_stderr: list,
                  list_ratio_otf: list,
                  list_best_solution_history: list, list_stderr_history: list,
                  list_time_for_optimization: list,
                  list_all_prior_tf: list, list_all_prior_stderr: list,
                  list_time_acquisition_optimization: list = None,
                  list_time_update_surrogate: list = None,
                  path_experiment: str = '') -> None:
    """
    Saves the raw data of the optimization process.
    @param list_time_update_surrogate:
    @param list_time_acquisition_optimization:
    @param list_all_prior_stderr:
    @param list_all_prior_tf:
    @param list_best_otf:
    @param list_best_otf_stderr:
    @param list_baseline_otf:
    @param list_baseline_otf_stderr:
    @param list_ratio_otf:
    @param list_best_solution_history:
    @param list_stderr_history:
    @param list_time_for_optimization:
    @param path_experiment:
    """

    if not os.path.isdir(path_experiment):
        os.makedirs(path_experiment)
    # turn lists into numpy arrays and save them in path_experiment
    np.save(os.path.join(path_experiment, 'list_best_otf'), np.array(list_best_otf))
    np.save(os.path.join(path_experiment, 'list_best_otf_stderr'), np.array(list_best_otf_stderr))
    np.save(os.path.join(path_experiment, 'list_baseline_otf'), np.array(list_baseline_otf))
    np.save(os.path.join(path_experiment, 'list_baseline_otf_stderr'), np.array(list_baseline_otf_stderr))
    np.save(os.path.join(path_experiment, 'list_ratio_otf'), np.array(list_ratio_otf))
    np.save(os.path.join(path_experiment, 'list_best_solution_history'), np.array(list_best_solution_history))
    np.save(os.path.join(path_experiment, 'list_stderr_history'), np.array(list_stderr_history))
    np.save(os.path.join(path_experiment, 'list_time_for_optimization'), np.array(list_time_for_optimization))
    np.save(os.path.join(path_experiment, 'list_all_prior_tf'), list_all_prior_tf)
    np.save(os.path.join(path_experiment, 'list_all_prior_stderr'), list_all_prior_stderr)
    if list_time_acquisition_optimization is not None:
        np.save(os.path.join(path_experiment, 'list_time_acquisition_optimization'), list_time_acquisition_optimization)
    if list_time_update_surrogate is not None:
        np.save(os.path.join(path_experiment, 'list_time_update_surrogate'), list_time_update_surrogate)


def load_raw_data(path_experiment: str) -> dict:
    """

    @param path_experiment:
    @return:
    """
    list_best_otf = np.load(os.path.join(path_experiment, 'list_best_otf.npy'))
    list_best_otf_stderr = np.load(os.path.join(path_experiment, 'list_best_otf_stderr.npy'))
    list_baseline_otf = np.load(os.path.join(path_experiment, 'list_baseline_otf.npy'))
    list_baseline_otf_stderr = np.load(os.path.join(path_experiment, 'list_baseline_otf_stderr.npy'))
    list_ratio_otf = np.load(os.path.join(path_experiment, 'list_ratio_otf.npy'))
    list_best_solution_history = np.load(os.path.join(path_experiment, 'list_best_solution_history.npy'))
    list_stderr_history = np.load(os.path.join(path_experiment, 'list_stderr_history.npy'))
    list_time_for_optimization = np.load(os.path.join(path_experiment, 'list_time_for_optimization.npy'))
    list_all_prior_tf = np.load(os.path.join(path_experiment, 'list_all_prior_tf.npy'))
    list_all_prior_stderr = np.load(os.path.join(path_experiment, 'list_all_prior_stderr.npy'), )

    # TODO this is a temporary fix, because old experiments do not have these files
    if os.path.isfile(os.path.join(path_experiment, 'list_time_acquisition_optimization.npy')) \
            and os.path.isfile(os.path.join(path_experiment, 'list_time_update_surrogate.npy')):
        list_time_acquisition_optimization = np.load(os.path.join(path_experiment, 'list_time_acquisition_optimization.npy'))
        list_time_update_surrogate = np.load(os.path.join(path_experiment, 'list_time_update_surrogate.npy'))
        return {'list_best_otf': list_best_otf, 'list_best_otf_stderr': list_best_otf_stderr,
                'list_baseline_otf': list_baseline_otf, 'list_baseline_otf_stderr': list_baseline_otf_stderr,
                'list_ratio_otf': list_ratio_otf,
                'list_best_solution_history': list_best_solution_history, 'list_stderr_history': list_stderr_history,
                'list_time_for_optimization': list_time_for_optimization,
                'list_all_prior_tf': list_all_prior_tf, 'list_all_prior_stderr': list_all_prior_stderr,
                'list_time_acquisition_optimization': list_time_acquisition_optimization,
                'list_time_update_surrogate': list_time_update_surrogate}
    else:

        return {'list_best_otf': list_best_otf, 'list_best_otf_stderr': list_best_otf_stderr,
                'list_baseline_otf': list_baseline_otf, 'list_baseline_otf_stderr': list_baseline_otf_stderr,
                'list_ratio_otf': list_ratio_otf,
                'list_best_solution_history': list_best_solution_history, 'list_stderr_history': list_stderr_history,
                'list_time_for_optimization': list_time_for_optimization,
                'list_all_prior_tf': list_all_prior_tf, 'list_all_prior_stderr': list_all_prior_stderr}


def choose_sentinels(node_attributes: list,
                     sentinels: int,
                     mode: str) -> list:
    """
    Chooses the sentinels with the highest attribute for the given mode.
    The first item in the list is always the highest of the chosen attribute
    node_attributes is a list to allow the dynamic use of the function in the main script.
    @param node_attributes: list of node attributes (degrees, capacity, transmission)
    @param sentinels: int, number of sentinels to choose
    @param mode: string, attribute to choose sentinels for
    @return: list of indices of the chosen sentinels sorted by mode
    """
    if mode == 'degree':
        node_degrees = node_attributes[0]
        # sort list of nodes by degree and return their indices
        nodes_degrees_sorted = sorted(node_degrees, key=lambda node_degrees: node_degrees[1], reverse=True)
        indices_highest_degree_nodes = [i[0] for i in nodes_degrees_sorted]
        # return only the n sentinels
        return indices_highest_degree_nodes[:sentinels]
    elif mode == 'capacity':
        node_capacities = node_attributes[1]
        node_capacities = [(i, c) for i, c in enumerate(node_capacities)]
        # sort list of nodes by capacity and get only their indices
        node_capacities_sorted = sorted(node_capacities, key=lambda node_capacities: node_capacities[1], reverse=True)
        indices_highest_capacity_nodes = [i[0] for i in node_capacities_sorted]

        return indices_highest_capacity_nodes[:sentinels]
    elif mode == 'transmission':
        node_transmissions = node_attributes[2]
        raise NotImplementedError
    else:
        raise ValueError(f'Mode "{mode}" not supported')


def get_node_attributes(node_attributes: list,
                        mode: str) -> list:
    """
    Returns the node attributes for the given mode.
    The node attributes are sorted by their
    node_attributes is a list to allow the dynamic use of the function in the main script.
    @param node_attributes: list of all node attributes (degrees, capacity, transmission)
    @param mode: string, attribute to return
    @return: list of node_attributes sorted by index
    """
    if mode == 'degree':
        node_degrees = node_attributes[0]
        # sort list of nodes by degree and return their indices
        nodes_degrees_sorted = sorted(node_degrees, key=lambda node_degrees: node_degrees[0])
        nodes_degrees_sorted = [i[1] for i in nodes_degrees_sorted]
        return nodes_degrees_sorted
    elif mode == 'capacity':
        node_capacities = node_attributes[1]
        node_capacities = [(i, c) for i, c in enumerate(node_capacities)]
        # sort list of nodes by capacity and get only their indices
        node_capacities_sorted = sorted(node_capacities, key=lambda node_capacities: node_capacities[0])
        node_capacities_sorted = [i[1] for i in node_capacities_sorted]

        return node_capacities_sorted
    elif mode == 'transmission':
        node_transmissions = node_attributes[2]
        raise NotImplementedError
    else:
        raise ValueError(f'Mode "{mode}" not supported')


def map_low_dim_x_to_high_dim(x: np.array,
                              number_of_nodes: int,
                              node_indices: list) -> np.array:
    """
    Map the values in an array x to an empty array of size n_nodes. The location of each xi is based on node_indices.
    @param x: numpy array, output array to be extended to size n_nodes
    @param number_of_nodes: int, defines size of output array
    @param node_indices: list of integers
    @return: numpy array of size n_node, with values of x at indices node_indices
    """
    # create a dummy vector to be filled with the values of x at the appropriate indices
    x_true = np.zeros(number_of_nodes)
    for i, xi in zip(node_indices, x):
        x_true[i] = xi
    return x_true


def create_test_strategy_prior(n_nodes: int,
                               node_degrees: list,
                               node_capacities: list,
                               total_budget: float,
                               sentinels: int,
                               mixed_strategies: bool = True,
                               only_baseline: bool = False) -> tuple:
    """
    Creates a list of test strategies to be used as a prior.
    First element in the list is a strategy where the budget is uniformly distributed over all sentinel nodes
    that the objective function is using.
    @param only_baseline: bool, sets whether only the baseline strategy is used as a prior
    @param mixed_strategies: bool, sets whether strategies with a mix of high degree and high capacity nodes are used
    @param sentinels:
    @param n_nodes: int, number of nodes in the SI-simulation graph
    @param node_degrees: list, contains indices of nodes and their degree
    @param node_capacities: list, capacity of each node
    @param total_budget: float, total budget for the allocation
    @return: three lists,
        1. list of numpy arrays, each array defining the budget distribution over a number of sentinels
        2. list of node indices for the sentinels in each strategy
        3. a string containing descriptions for each test strategy
    """
    prior_test_strategies = []
    prior_node_indices = []
    test_strategy_parameter = 'number\tdescription'     # string containing descriptions for each test strategy

    # specify increasing number of sentinels TODO why these numbers in specific ?
    # choose new sentinel nodes from the original number of sentinels
    sentinels_list = [int(sentinels / 6), int(sentinels / 12), int(sentinels / 24)]

    # get the (sorted) indices of the highest degree nodes. A list of length sentinels
    indices_highest_degree_nodes = choose_sentinels(node_attributes=[node_degrees, None, None],
                                                    sentinels=sentinels,
                                                    mode='degree')

    indices_highest_capacity_nodes = choose_sentinels(node_attributes=[None, node_capacities, None],
                                                      sentinels=sentinels,
                                                      mode='capacity')

    # ------------------------------------------------------------------------
    # the uniform distribution of the total budget over all sentinels
    x_sentinels = np.array([total_budget / sentinels for _ in range(sentinels)])
    prior_test_strategies.append(x_sentinels)
    prior_node_indices.append(indices_highest_degree_nodes)

    test_strategy_parameter += f'\n0\tuniform distribution over all {n_nodes} nodes'

    if not only_baseline:
        n = 1   # index for test strategies
        for i, s in enumerate(sentinels_list):
            # TODO for this to work I need a new indices list of the s-highest degree nodes of sentinels
            # for the strategies with sentinels from the sentinels list (budget is distributed among less nodes than
            # sentinels), choose the s highest ones and create simple indices for them.
            # before appending them to the prior, the new strategy is mapped to the dimension of the sentinels.
            # If this is not done the strategies in the prior will not work with the optimizers.
            idx = list(range(len(indices_highest_degree_nodes[:s])))
            # ------
            # create strategy for s highest degree nodes, budget is allocated uniformly
            x_sentinels = np.array([total_budget / s for _ in range(s)])
            # map the new strategy to the original number of sentinels
            new_strategy = map_low_dim_x_to_high_dim(x=x_sentinels,
                                                     number_of_nodes=sentinels,
                                                     node_indices=idx)
            prior_test_strategies.append(new_strategy)
            prior_node_indices.append(indices_highest_degree_nodes)

            test_strategy_parameter += f'\n{n}\tuniform distribution over {s} highest degree nodes'
            n += 1
            # ------
            # create strategy for s highest capacity nodes, budget is allocated uniformly

            x_sentinels = np.array([total_budget / s for _ in range(s)])
            new_strategy = map_low_dim_x_to_high_dim(x=x_sentinels,
                                                     number_of_nodes=sentinels,
                                                     node_indices=idx)
            prior_test_strategies.append(new_strategy)
            prior_node_indices.append(indices_highest_capacity_nodes)

            test_strategy_parameter += f'\n{n}\tuniform distribution over {s} highest capacity nodes'
            n += 1
            # TODO the mixed priors don't work if sentinels < n_nodes
            # in that case it is not sufficient to just take the first n sentinels as indices
            # # ------
            # # create strategies that are a mix of the highest degree and highest capacity nodes
            # if mixed_strategies:
            #     for k in range(s)[1:]:
            #
            #         # get the highest degree nodes and highest capacity nodes and
            #         # check whether node indices would appear twice and remove the duplicates
            #         # TODO maybe there is a better method for this check??
            #         indices_combined = list(set(indices_highest_degree_nodes[:k]) |
            #                                 set(indices_highest_capacity_nodes[:s-k]))
            #
            #         idx = list(range(len(indices_combined)))
            #
            #         print('len indices and sentinels', len(indices_combined), sentinels)
            #         # because of the missing nodes the strategies might violate the sum constraint (lightly)
            #         # therefore of this the allocated budget is smaller or greater than the total budget
            #         x_sentinels = np.array([total_budget / len(indices_combined) for _ in indices_combined])
            #         print('x_sentinels', np.shape(x_sentinels))
            #         print('stats0: ', np.min(x_sentinels), np.max(x_sentinels), np.mean(x_sentinels), np.sum(x_sentinels))
            #         print('budget', total_budget, total_budget / len(indices_combined))
            #         new_strategy = map_low_dim_x_to_high_dim(x=x_sentinels,
            #                                                  number_of_nodes=sentinels,
            #                                                  node_indices=idx)
            #         print('new_strategy', np.shape(new_strategy))
            #         print('stats', np.min(new_strategy), np.max(new_strategy), np.mean(new_strategy), np.sum(new_strategy), '\n')
            #         prior_test_strategies.append(new_strategy)
            #         prior_node_indices.append(indices_combined)
            #
            #         test_strategy_parameter += f'\n{n}\tuniform distribution over ' \
            #                                    f'{k} highest degree nodes and {s-k} highest capacity nodes'
            #         n += 1

    return prior_test_strategies, prior_node_indices, test_strategy_parameter


def baseline(total_budget: float,
             eval_function: callable,
             n_nodes: int,
             parallel: bool,
             num_cpu_cores: int,
             statistic: callable) -> tuple:
    """
    Creates a test strategy where the total budget is uniformly allocated to all nodes.
    Evaluates the test strategy with the given evaluation function for 10000 simulation runs.
    @param statistic: function, aggregates the results of the simulation runs
    @param total_budget: int, total budget for the test strategy
    @param eval_function: SI-simulation
    @param n_nodes: int, number of nodes in the network
    @param parallel: bool, whether to use parallelization for the SI-simulation
    @param num_cpu_cores: int, number of cpu cores to use for parallelization
    @return: mean and standard error of uniform baseline and the corresponding uniform test strategy
    """
    # distribute budget uniformly over all nodes
    x_baseline = np.array([total_budget / n_nodes for _ in range(n_nodes)])

    m, stderr = eval_function(x_baseline,
                              n_simulations=10000,
                              parallel=parallel,
                              num_cpu_cores=num_cpu_cores,
                              statistic=statistic
                              )

    return m, stderr, x_baseline


def test_function(x: np.array, *args, **kwargs) -> float:
    """
    Quadratic function just for test purposes
    @param x: numpy array, input vector
    @return: float, square of first element of input vector
    """
    return x[0]**2


def evaluate_prior(prior: list,
                   n_simulations: int,
                   eval_function: callable,
                   parallel: bool,
                   num_cpu_cores: int,
                   statistic: callable) -> np.array:
    """
    Evaluate the strategies in the prior and return the mean and standard error
    @param statistic:
    @param prior: list of numpy arrays, each array contains the values of a test strategy
    @param n_simulations: int, number of simulations to be performed
    @param eval_function: function, function to be evaluated
    @param parallel: bool, whether to use parallelization or not
    @param num_cpu_cores: int, number of cpu cores to be used for parallelization
    @return: A list of the mean and standard error for every strategy in the prior
    """
    y_prior = []
    for strategy in tqdm(prior, leave=False):
        m, stderr = eval_function(strategy,
                                  n_simulations=n_simulations,
                                  parallel=parallel,
                                  num_cpu_cores=num_cpu_cores,
                                  statistic=statistic)

        y_prior.append(np.array([m, stderr]))

    return np.array(y_prior)


def create_graph(n: int,
                 graph_type: str,
                 n_nodes: int,
                 base_path: str = '../data/') -> tuple:
    """
    Loads n_runs graphs from disk and returns them as a list
    @param n_nodes:
    @param base_path:
    @param n: int, the number of the graph to be loaded
    @param graph_type: string, barabasi-albert, waxman or synthetic
    @return: three lists, containing transmissions, capacities and degrees for the loaded graph respectively
            all lists are sorted by node index
    """
    if graph_type == 'waxman':
        network_path = os.path.join(base_path, graph_type + '_networks', f'{n_nodes}')
        # print(f'Loading waxman graph number {n}')

        transmission_path = os.path.join(network_path, f'WX{n}', 'transmissions.txt')
        transmissions_waxman = pd.read_csv(transmission_path, header=None).to_numpy()

        capacities_path = os.path.join(network_path, f'WX{n}', 'capacity.txt')
        capacities_waxman = np.int_(pd.read_csv(capacities_path, header=None).to_numpy().squeeze())
        # node_capacities contains only capacities, add node_index sort nodes by capacities and get their indices
        # capacities_waxman = [(i, c) for i, c in enumerate(capacities_waxman)]

        degrees_path = os.path.join(network_path, f'WX{n}', 'degree.txt')
        degrees_waxman = pd.read_csv(degrees_path, header=None).to_numpy()

        transmissions = transmissions_waxman
        capacities = capacities_waxman
        degrees = degrees_waxman

    elif graph_type == 'ba':
        network_path = os.path.join(base_path, graph_type + '_networks', f'{n_nodes}')
        # print(f'Loading barabasi-albert graph number {n}')

        single_transmission_path = os.path.join(network_path, f'BA{n}', 'transmissions.txt')
        transmissions_ba = pd.read_csv(single_transmission_path, header=None).to_numpy()

        capacities_path = os.path.join(network_path, f'BA{n}', 'capacity.txt')
        capacities_ba = np.int_(pd.read_csv(capacities_path, header=None).to_numpy().squeeze())
        # node_capacities contains only capacities, add node_index sort nodes by capacities and get their indices
        # capacities_ba = [(i, c) for i, c in enumerate(capacities_ba)]

        degrees_path = os.path.join(network_path, f'BA{n}', 'degree.txt')
        degrees_ba = pd.read_csv(degrees_path, header=None).to_numpy()

        transmissions = transmissions_ba
        capacities = capacities_ba
        degrees = degrees_ba

    elif graph_type == 'syn':
        network_path = os.path.join(base_path, f'Synset{n_nodes}-180')
        # print(f'Loading synthetic graph number {n}')

        transmissions_path = os.path.join(network_path, f'syndata{n}', 'dataset.txt')
        transmissions = pd.read_csv(transmissions_path, header=None)
        transmissions = transmissions[[2, 2, 0, 1, 3]]
        edge_list = transmissions[[0, 1]]
        transmissions_syn = transmissions.to_numpy()

        G = nx.from_pandas_edgelist(edge_list, source=0, target=1, create_using=nx.DiGraph())
        degrees_syn = np.array(list(G.degree()))

        # some nodes are slaughterhouses which have no trading partners, only incoming transmissions
        # Consequently, they don't exist in the network although they are needed for the simulation
        # they get assigned a degree of 0
        if np.shape(degrees_syn)[0] < n_nodes:
            # get the indices of the missing nodes
            all_node_indices = list(range(n_nodes))
            missing_node_indices = list(set(all_node_indices) - set(degrees_syn[:, 0]))
            # assign the missing nodes a degree of 0
            missing_degrees = np.array([[i, 0] for i in missing_node_indices])
            # join the missing degrees with the existing degrees
            degrees_syn = np.concatenate((degrees_syn, missing_degrees), axis=0)

        capacities_path = os.path.join(network_path, f'syndata{n}', 'barn_size.txt')
        capacities = pd.read_csv(capacities_path, header=None)
        capacities_syn = capacities.iloc[0][:n_nodes].to_numpy()
        # node_capacities contains only capacities, add node_index sort nodes by capacities and get their indices
        # capacities_syn = [(i, c) for i, c in enumerate(capacities_syn)]

        transmissions = transmissions_syn
        capacities = capacities_syn
        degrees = degrees_syn

    else:
        Exception(f'Graph type {graph_type} not supported')

    return transmissions, capacities, degrees


def rms_tia(n_infected_animals: np.array) -> tuple:
    values = n_infected_animals**2
    estimate = np.sqrt(np.mean(values, axis=0))
    stderr = np.std(values, ddof=1, axis=0) / np.sqrt(values.shape[0])
    stderr = stderr/(2*estimate)
    return estimate, stderr


def mean_tia(n_infected_animals: np.array) -> tuple:
    estimate = np.mean(n_infected_animals, axis=0)
    stderr = np.std(n_infected_animals, ddof=1, axis=0) / np.sqrt(n_infected_animals.shape[0])
    return estimate, stderr


def percentile_tia(n_infected_animals: np.array) -> tuple:
    estimate = np.percentile(n_infected_animals, 95, axis=0)
    stderr = mjci(n_infected_animals, prob=[0.95], axis=0)[0]
    return estimate, stderr
