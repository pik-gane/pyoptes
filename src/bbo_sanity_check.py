'''
Computes the average target function output for the uniform baseline.
'''


from pyoptes.optimization.budget_allocation import target_function as f
from pyoptes import create_graph, compute_average_otf_and_stderr

import argparse
import numpy as np
from tqdm import tqdm
from scipy.stats.mstats import mjci


def rms_tia(n_infected_animals):
    values = n_infected_animals**2
    estimate = np.sqrt(np.mean(values, axis=0))
    stderr = np.std(values, ddof=1, axis=0) / np.sqrt(values.shape[0])
    stderr = stderr/(2*estimate)
    return estimate, stderr


def mean_tia(n_infected_animals):
    estimate = np.mean(n_infected_animals, axis=0)
    stderr = np.std(n_infected_animals, ddof=1, axis=0) / np.sqrt(n_infected_animals.shape[0])
    return estimate, stderr


def percentile_tia(n_infected_animals):
    estimate = np.percentile(n_infected_animals, 95, axis=0)
    stderr = mjci(n_infected_animals, prob=[0.95], axis=0)[0]
    return estimate, stderr


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--n_nodes", type=int, default=1040,
                        help="Si-simulation parameter. "
                             "Defines the number of nodes used by the SI-model to create a graph. "
                             "Default value is 120 nodes.")

    parser.add_argument('--n_runs', type=int, default=100,
                        help='')
    parser.add_argument("--n_simulations", type=int, default=10000,
                        help="Si-simulation parameter. Sets the number of runs the for the SI-model. "
                             "Higher values of n_simulations lower the variance of the output of the simulation. "
                             "Default value is 1000.")
    parser.add_argument('--graph_type', choices=['waxman', 'ba', 'syn'], default='syn',
                        help='Si-simulation parameter. Set the type of graph the simulation uses.'
                             ' Either Waxman or Barabasi-Albert (ba) can be used. Default is Barabasi-Albert.')

    parser.add_argument('--scale_total_budget', type=float, default=1.0,
                        help="SI-simulation parameter. Scales the total budget for SI-model. Default is 1.0.")
    parser.add_argument('--path_networks', default='../../networks/data',
                        help='Location where the networks are saved to. '
                             'Path on cluster. /p/projects/ou/labs/gane/optes/mcmc_100nets/data'
                             '/p/projects/ou/labs/gane/optes/mcmc_100nets/data/')
    args = parser.parse_args()

    total_budget = args.scale_total_budget * args.n_nodes

    list_baseline_otf = []  # baseline  function value on each network and corresponding standard error
    list_baseline_otf_stderr = []

    for n in tqdm(range(args.n_runs)):
        transmissions, capacities, degrees = create_graph(n, args.graph_type, args.n_nodes, args.path_networks)

        f.prepare(n_nodes=args.n_nodes,
                  capacity_distribution=capacities,
                  pre_transmissions=transmissions,
                  p_infection_by_transmission=0.5,
                  delta_t_symptoms=60,
                  expected_time_of_first_infection=30,
                  static_network=None,
                  use_real_data=False)

        x_baseline = np.array([total_budget / args.n_nodes for _ in range(args.n_nodes)])

        m, stderr = f.evaluate(x_baseline,
                               n_simulations=10000,
                               parallel=True,
                               num_cpu_cores=32,
                               statistic=rms_tia)

        list_baseline_otf.append(m)
        list_baseline_otf_stderr.append(stderr)
    print(list_baseline_otf)
    average_baseline, average_baseline_stderr = compute_average_otf_and_stderr(list_baseline_otf,
                                                                               list_baseline_otf_stderr,
                                                                               n_runs=args.n_runs)

    print('average_baseline', average_baseline)
    print('average_baseline_stderr', average_baseline_stderr)