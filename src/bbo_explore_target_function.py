'''
Compute the target function value for different number of sentinels (either highest degree, transmissions, capacities)

'''

from pyoptes.optimization.budget_allocation import target_function as f
from pyoptes import create_graph, compute_average_otf_and_stderr, map_low_dim_x_to_high_dim
from pyoptes import choose_sentinels
from pyoptes import rms_tia, percentile_tia, mean_tia
from pyoptes import plot_effect_of_different_sentinels

import argparse
import numpy as np
from tqdm import tqdm
from scipy.stats.mstats import mjci

# TODO look into maltes file in the repo
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # parser.add_argument()
    parser.add_argument('--n_runs', type=int, default=100,
                        help='')
    parser.add_argument('--sentinels_step_size', type=int, default=5,
                        help='Sets the step size for the creation of the ....')

    # ------------------ SI-simulation hyperparameters ------------------
    parser.add_argument("--statistic", choices=['mean', 'rms', '95perc'], default='rms',
                        help="Choose the statistic to be used by the target function. "
                             "Choose between mean, rms (root-mean-square) or 95perc (95th-percentile).")
    parser.add_argument("--n_simulations", type=int, default=10000,
                        help="Si-simulation parameter. Sets the number of runs the for the SI-model. "
                             "Higher values of n_simulations lower the variance of the output of the simulation. "
                             "Default value is 1000.")
    parser.add_argument('--graph_type', choices=['waxman', 'ba', 'syn'], default='syn',
                        help='Si-simulation parameter. Set the type of graph the simulation uses.'
                             ' Either Waxman,Synthetic or Barabasi-Albert (ba) can be used. Default is Synthetic.')
    parser.add_argument('--scale_total_budget', type=int, default=1, choices=[1, 4, 12],
                        help="SI-simulation parameter. Scales the total budget for SI-model. Default is 1.")
    parser.add_argument('--parallel', type=bool, default=True,
                        help='Si-simulation parameter. Sets whether multiple simulations run are to be done in parallel'
                             'or sequentially. Default is set to parallel computation.')
    parser.add_argument("--num_cpu_cores", type=int, default=32,
                        help='Si-simulation parameter. Defines the number of cpus to be used for the simulation '
                             'parallelization. If more cpus are chosen than available, the max available are selected.'
                             '-1 selects all available cpus. Default are 32 cpus.')

    parser.add_argument('--delta_t_symptoms', type=int, default=60,
                        help='Si-simulation parameter.. Sets the time (in days) after which an infection is detected'
                             ' automatically. Default is 60 days')
    parser.add_argument('--p_infection_by_transmission', type=float, default=0.5,
                        help='Si-simulation parameter. The probability of how likely a trade animal '
                             'infects other animals. Default is 0.5.')
    parser.add_argument('--expected_time_of_first_infection', type=int, default=30,
                        help='Si-simulation parameter. The expected time (in days) after which the first infection occurs. ')

    # ------------------ utility hyperparameters ------------------
    parser.add_argument('--mode_choose_sentinels', choices=['degree', 'capacity', 'transmission'], default='degree',
                        help='Sets the mode of how sentinels are chosen. ')
    parser.add_argument('--path_networks', default='../../networks/data',
                        help='Location where the networks are saved to. '
                             'Path on cluster. /p/projects/ou/labs/gane/optes/mcmc_100nets/data'
                             '/p/projects/ou/labs/gane/optes/mcmc_100nets/data/')
    args = parser.parse_args()

    #
    nodes = [120, 1040, 57590]

    # define function to average the results of the simulation
    if args.statistic == 'mean':
        statistic = mean_tia
    elif args.statistic == 'rms':
        statistic = rms_tia
    elif args.statistic == '95perc':
        statistic = percentile_tia
    else:
        raise ValueError('Statistic not supported')

    for n in nodes:
        print(f'Running simulation with {n} nodes')
        # create a list of sentinels from 0 to all sentinels
        sentinels = list(range(0, n+5, 5)) #TODO make 5 a parameter

        total_budget = n

        list_all_m = []
        list_all_stderr = []

        # compute the target function value for different number of sentinels
        for s in tqdm(sentinels):
            list_m = []
            list_stderr = []
            for run in tqdm(range(args.n_runs), leave=False):

                # create graph and prepare the simulation
                transmissions, capacities, degrees = create_graph(n=run,
                                                                  graph_type=args.graph_type,
                                                                  n_nodes=n,
                                                                  base_path=args.path_networks)

                f.prepare(n_nodes=n,
                          capacity_distribution=capacities,
                          pre_transmissions=transmissions,
                          p_infection_by_transmission=args.p_infection_by_transmission,
                          delta_t_symptoms=args.delta_t_symptoms,
                          expected_time_of_first_infection=args.expected_time_of_first_infection,
                          static_network=None,
                          use_real_data=False)

                sentinel_indices = choose_sentinels([degrees, None, None], s, args.mode_choose_sentinels)

                # create a budget vector
                budget = np.ones(s)*total_budget/s

                budget = map_low_dim_x_to_high_dim(budget, n, sentinel_indices)

                m, stderr = f.evaluate(budget_allocation=budget,
                                       n_simulations=args.n_simulations,
                                       parallel=args.parallel,
                                       num_cpu_cores=args.num_cpu_cores,
                                       statistic=statistic)

                list_m.append(m)
                list_stderr.append(stderr)
            # average run results
            mm, m_stderr = compute_average_otf_and_stderr(list_m, list_stderr, args.n_runs)
            # save the averaged results
            list_all_m.append(mm)
            list_all_stderr.append(m_stderr)
        # create a bar plot from mean, stderr pair for each sentinel number
        # x-axis is number of sentinels, y are mean and stderr

        # TODO maybe create additional plots with capacity sentinels and other attributes
        plot_effect_of_different_sentinels(number_of_sentinels=sentinels,
                                           m=list_all_m,
                                           stderr=list_all_stderr,
                                           path_experiment='',
                                           title=f'Simulations with increasing number of sentinels. {n} nodes',
                                           n_nodes=n)