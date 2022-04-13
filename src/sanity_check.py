import os.path
import pylab as plt

from pyoptes.optimization.budget_allocation import target_function as f

import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd
from scipy.stats import lognorm


def create_graphs(n_runs, graph_type, network_path):
    """
    Loads n_runs graphs from disk and returns them as a list
    @param n_runs: int, the number of different networks to be loaded
    @param graph_type: string, barabasi-albert or waxman
    @return: list of dictionaries with the graph and the node indices
    """
    assert 0 < n_runs <= 100    # there are only 100 graphs available
    network_list = []

    if graph_type == 'waxman':
        print(f'Loading {n_runs} waxman graphs')
        for n in tqdm(range(n_runs)):
            transmission_path = os.path.join(network_path, f'WX{n}', 'transmissions.txt')
            transmissions_waxman = pd.read_csv(transmission_path, header=None).to_numpy()

            capacities_path = os.path.join(network_path, f'WX{n}', 'capacity.txt')
            capacities_waxman = pd.read_csv(capacities_path, header=None).to_numpy().squeeze()

            degrees_path = os.path.join(network_path, f'WX{n}', 'degree.txt')
            degrees_waxman = pd.read_csv(degrees_path, header=None).to_numpy()

            network_list.append([transmissions_waxman,
                                 np.int_(capacities_waxman),
                                 degrees_waxman])
    elif graph_type == 'ba':
        print(f'Loading {n_runs} barabasi-albert graphs')
        for n in tqdm(range(n_runs)):
            single_transmission_path = os.path.join(network_path, f'BA{n}', 'transmissions.txt')
            transmissions_ba = pd.read_csv(single_transmission_path, header=None).to_numpy()

            capacities_path = os.path.join(network_path, f'BA{n}', 'capacity.txt')
            capacities_ba = pd.read_csv(capacities_path, header=None).to_numpy().squeeze()

            degrees_path = os.path.join(network_path, f'BA{n}', 'degree.txt')
            degrees_ba = pd.read_csv(degrees_path, header=None).to_numpy()

            network_list.append([transmissions_ba,
                                 np.int_(capacities_ba),
                                 degrees_ba])
    else:
        Exception(f'Graph type {graph_type} not supported')

    return network_list


def est_prob_and_stderr(is_infected):
    ps = np.mean(is_infected, axis=0)
    stderrs = np.sqrt(ps * (1 - ps) / is_infected.shape[0])
    return ps, stderrs


def n_infected_animals(is_infected):
    return np.sum(f.capacities * is_infected)


def mean_square_and_stderr(n_infected_animals):
    values = n_infected_animals ** 2
    estimate = np.mean(values, axis=0)
    stderr = np.std(values, ddof=1, axis=0) / np.sqrt(values.shape[0])
    return estimate, stderr


def caps(size):
    return lognorm.rvs(s=2, scale=np.exp(4), size=size)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--n_nodes", type=int, default=120,
                        help="Si-simulation parameter. "
                             "Defines the number of nodes used by the SI-model to create a graph. "
                             "Default value is 120 nodes.")

    parser.add_argument('--n_runs', type=int, default=100,
                        help='')
    parser.add_argument("--n_simulations", type=int, default=100000,
                        help="Si-simulation parameter. Sets the number of runs the for the SI-model. "
                             "Higher values of n_simulations lower the variance of the output of the simulation. "
                             "Default value is 1000.")
    parser.add_argument('--graph_type', choices=['waxman', 'ba'], default='ba',
                        help='Si-simulation parameter. Set the type of graph the simulation uses.'
                             ' Either Waxman or Barabasi-Albert (ba) can be used. Default is Barabasi-Albert.')
    parser.add_argument('--delta_t_symptoms', type=int, default=60,
                        help='Si-simulation parameter.. Sets the time (in days) after which an infection is detected'
                             ' automatically. Default is 60 days')
    parser.add_argument('--p_infection_by_transmission', type=float, default=0.5,
                        help='Si-simulation parameter. The probability of how likely a trade animal '
                             'infects other animals. Default is 0.5.')

    parser.add_argument('--scale_total_budget', type=float, default=1.0,
                        help="SI-simulation parameter. Scales the total budget for SI-model. Default is 1.0.")
    args = parser.parse_args()

    total_budget = args.scale_total_budget * args.n_nodes

    # change '../data' to the path where you saved the networks
    network_path = os.path.join('../data/', args.graph_type + '_networks', f'{args.n_nodes}')
    # creates a list of n_runs networks (either waxman or barabasi-albert)
    network_list = create_graphs(args.n_runs, args.graph_type, network_path)

    list_baseline_mean = []
    list_baseline_stderr = []

    max_degree = []

    mean_sq_evaluation_params = {
        'aggregation': n_infected_animals,
        'statistic': mean_square_and_stderr,
        'n_simulations': args.n_simulations,
        'parallel': True,
        'num_cpu_cores': -1
    }

    for n, network in tqdm(enumerate(network_list[:args.n_runs])):
        # unpack the properties of the network

        transmissions, capacities, degree = network

        f.prepare(n_nodes=args.n_nodes,
                  delta_t_symptoms=args.delta_t_symptoms,
                  p_infection_by_transmission=args.p_infection_by_transmission,
                  static_network=None,
                  capacity_distribution=capacities,
                  pre_transmissions=transmissions,
                  max_t=365)

        # distribute budget uniformly over all nodes
        x_baseline = np.array([total_budget / args.n_nodes for _ in range(args.n_nodes)])

        # run the optimization

        baseline_mean, baseline_stderr = f.evaluate(x_baseline, **mean_sq_evaluation_params)

        rms_baseline = np.sqrt(baseline_mean)
        stderr_baseline = baseline_stderr / (2*rms_baseline)

        list_baseline_mean.append(rms_baseline)
        list_baseline_stderr.append(stderr_baseline)

        max_degree.append(np.max(degree[:, 1]))

    print('ratio of stderr and mean:', np.mean(list_baseline_stderr) / np.mean(list_baseline_mean))
    average_baseline_mean = np.mean(list_baseline_mean)
    average_baseline_stderr = np.mean(list_baseline_stderr)
    # average_best_test_strategy = np.mean(list_best_test_strategy, axis=0)
    print(f'average_baseline_mean: {average_baseline_mean},'
          f' average_baseline_stderr: {average_baseline_stderr}'
          f'ratio stderr/mean{average_baseline_stderr/average_baseline_mean}')

    baseline_rms = np.sqrt(np.array(list_baseline_mean))
    baseline_rms_stderr = np.array(list_baseline_stderr) / (2*baseline_rms)
    su = [list_baseline_mean[i] - list_baseline_stderr[i] for i in range(len(list_baseline_mean))]
    sd = [list_baseline_mean[i] + list_baseline_stderr[i] for i in range(len(list_baseline_mean))]

    average = np.ones(len(list_baseline_mean)) * average_baseline_mean
    plt.plot(range(len(list_baseline_mean)), list_baseline_mean, label='baseline')
    plt.plot(range(len(list_baseline_mean)), average, label='average baseline')
    # add standard error of the baseline
    plt.plot(range(len(list_baseline_mean)), su,
             label='stderr baseline', linestyle='dotted', color='red')
    plt.plot(range(len(list_baseline_mean)), sd,
             linestyle='dotted', color='red')

    plt.title(f'Baseline mean & stderr over {args.n_runs} runs')
    plt.xlabel('Iteration')
    plt.ylabel('SI-model output (rms)')
    plt.legend()
    plt.savefig('Baseline_mean_over_100_networks.png')
    plt.show()

    plt.clf()
    plt.scatter(max_degree, list_baseline_mean)
    plt.title(f'Highest degree vs baseline mean over {args.n_runs} runs')
    plt.xlabel('Highest degree')
    plt.ylabel('Baseline mean (rms)')
    plt.savefig('Scatter-plot_node_degree_vs_baseline.png')

    plt.show()
