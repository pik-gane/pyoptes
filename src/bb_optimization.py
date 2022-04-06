import os.path
import pylab as plt

from pyoptes.optimization.budget_allocation import target_function as f

from pyoptes import bo_cma, bo_pyGPGO

from pyoptes import choose_high_degree_nodes, baseline
from pyoptes import map_low_dim_x_to_high_dim, test_function, create_test_strategy_prior
from pyoptes import save_hyperparameters, save_results, plot_prior, create_graphs
from pyoptes import plot_time_for_optimization, plot_optimizer_history, evaluate_prior

import inspect
import argparse
import numpy as np
from tqdm import tqdm
from scipy.stats import lognorm
from time import time, localtime, strftime


# TODO what is this exactly ??
# TODO rename to something more meaningful
def caps(size):
    return lognorm.rvs(s=2, scale=np.exp(4), size=size)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("optimizer", choices=['cma', 'gpgo'],
                        help="Choose the optimizer to run on the SI-model. Choose between CMA-ES and GPGO")
    parser.add_argument("name_experiment", help="The name of the folder where the results of the optimizer run are"
                                                "saved to.")

    parser.add_argument("--sentinels", type=int, default=120,
                        help="Set the number of nodes that are used. Has to be smaller than or equal to n_nodes. "
                             "Default is 10 nodes.")
    parser.add_argument("--n_nodes", type=int, default=120,
                        help="Si-simulation parameter. "
                             "Defines the number of nodes used by the SI-model to create a graph. "
                             "Default value is 120 nodes.")

    parser.add_argument('--test_strategy_initialisation', choices=['uniform', 'random'], default='uniform',
                        help="Defines how the initial test strategy is initialised.")
    parser.add_argument('--n_runs', type=int, default=10,
                        help='')
    parser.add_argument("--n_simulations", type=int, default=1000,
                        help="Si-simulation parameter. Sets the number of runs the for the SI-model. "
                             "Higher values of n_simulations lower the variance of the output of the simulation. "
                             "Default value is 1000.")
    # TODO rename graph to something more clear ??
    parser.add_argument('--graph_type', choices=['waxman', 'ba'], default='ba',
                        help='Si-simulation parameter. Set the type of graph the simulation uses.'
                             ' Either Waxman or Barabasi-Albert (ba) can be used. Default is Barabasi-Albert.')
    parser.add_argument('--delta_t_symptoms', type=int, default=60,
                        help='Si-simulation parameter.. Sets the time (in days) after which an infection is detected'
                             ' automatically. Default is 60 days')
    parser.add_argument('--p_infection_by_transmission', type=float, default=0.5,
                        help='Si-simulation parameter. The probability of how likely a trade animal '
                             'infects other animals. Default is 0.5.')
    parser.add_argument('--parallel', type=bool, default=True,
                        help='Si-simulation parameter. Sets whether multiple simulations run are to be done in parallel'
                             'or sequentially. Default is set to parallel computation.')
    parser.add_argument("--cpu_count", type=int, default=14,
                        help='Si-simulation parameter. Defines the number of cpus to be used for the simulation '
                             'parallelization. If more cpus are chosen than available, the max available are selected.'
                             '-1 selects all available cpus. Default are 14 cpus.')

    parser.add_argument("--max_iterations", type=int, default=1000,
                        help="Optimizer parameter. The maximum number of iterations the algorithms run.")
    parser.add_argument('--cma_sigma', type=float, default=30,
                        help="Optimizer parameter. Defines the variance in objective function parameters "
                             "from which new population is sampled. Therefore the variance has to be big enough to"
                             "change the parameters in a meaningful way. A useful heuristic is to set the variance to "
                             "about 1/4th of the parameter search space. Default value (for 120 nodes) is 30.")
    parser.add_argument('--acquisition_function', default='EI',
                        choices=['EI', 'PI', 'UCB', 'Entropy', 'tEI'],
                        help='GPGO optimizer parameter. Defines the acquisition function that is used by GPGO.')
    parser.add_argument('--use_prior', type=bool, default=True,
                        help='GPGO optimizer parameter. Sets whether the surrogate function is fitted with priors '
                             'created by heuristics or by sampling random point. Only works when n_nodes and sentinels'
                             'are the same size. Default is True.')

    parser.add_argument('--plot_prior', type=bool, default='', help='')
    parser.add_argument("--log_level", type=int, default=3, choices=range(1, 11), metavar="[1-10]",
                        help="Optimizer parameter. Only effects SMAC and GPGO. Sets how often log messages appear. "
                             "Lower values mean more messages.")
    parser.add_argument('--path_plot', default='pyoptes/optimization/budget_allocation/blackbox_learning/plots/',
                        help="Optimizer parameter. Location where all the individual results"
                             " of the optimizers are saved to. "
                             "Default location is 'pyoptes/optimization/budget_allocation/blackbox_learning/plots/'")
    args = parser.parse_args()

    # prepare the directory for the plots, hyperparameters and results
    path_experiment = os.path.join(args.path_plot, args.name_experiment)
    if not os.path.exists(path_experiment):
        os.makedirs(path_experiment)

    #
    af = {'EI': 'ExpectedImprovement', 'PI': 'ProbabilityImprovement', 'UCB': 'UCB',
          'Entropy': 'Entropy', 'tEI': 'tExpectedImprovement'}
    acquisition_function = af[args.acquisition_function]

    # define function to average the results of the simulation
    # the mean of the squared ys is taken to emphasise the tail of the distribution of y
    statistic = lambda x: np.mean(x**2, axis=0)

    total_budget = 1.0 * args.n_nodes  # i.e., on average, nodes will do one test per year
    # define the first constraint, the boundaries of x_i
    bounds = [0, total_budget]

    # save SI-model and optimizer parameters as .json-file
    experiment_params = {'simulation_hyperparameters': {'total_budget': total_budget,
                                                        'n_nodes': args.n_nodes,
                                                        'graph': args.graph_type,
                                                        'sentinels': args.sentinels,
                                                        'n_simulations': args.n_simulations,
                                                        'statistic': inspect.getsourcelines(statistic)[0][0][23:-1],
                                                        'delta_t_symptoms': args.delta_t_symptoms,
                                                        'p_infection_by_transmission': args.p_infection_by_transmission,
                                                        'n_runs': args.n_runs
                                                        },
                         'optimizer_hyperparameters': {'optimizer': args.optimizer,
                                                       'max_iterations': args.max_iterations,
                                                       }}

    # creates a list of n_runs networks (either waxman or barabasi-albert)
    network_list = create_graphs(args.n_runs, args.graph_type, args.n_nodes)

    # TODO wrap the following in a loop over the network_list
    # TODO save optimizer output in a directory with structure: /name_experiment/runs/1-n_runs/...
    # TODO in addition save optimizer output in lists
    # TODO compute mean for baseline, best_parameter
    # TODO plot and save new averaged results

    if args.optimizer == 'cma':
        experiment_params['optimizer_hyperparameters']['cma_sigma'] = args.cma_sigma
    elif args.optimizer == 'gpgo':
        experiment_params['optimizer_hyperparameters']['use_prior'] = args.use_prior
        experiment_params['optimizer_hyperparameters']['acquisition_function'] = acquisition_function

    save_hyperparameters(experiment_params, path_experiment)

    list_prior = []
    list_baseline = []

    list_solution_history = []
    list_best_test_strategy = []
    list_time_for_optimization = []

    time_start = time()
    for n, network in enumerate(network_list[:args.n_runs]):

        # unpack the properties of the network
        transmissions, capacities, degrees = network

        f.prepare(n_nodes=args.n_nodes,
                  capacity_distribution=capacities,
                  p_infection_by_transmission=args.p_infection_by_transmission,
                  static_network=None,
                  delta_t_symptoms=args.delta_t_symptoms,
                  pre_transmissions=transmissions)

        # create a list of test strategies based on different heuristics
        prior, prior_parameter = create_test_strategy_prior(args.n_nodes, degrees,
                                                            capacities, total_budget, args.sentinels)
        list_prior.append(prior)

        # reduce the dimension of the input space by choosing to only allocate the budget between nodes with the highest
        # degrees. The function return the indices of these nodes
        # The indices correspond to the first item of the prior
        node_indices = choose_high_degree_nodes(degrees, args.sentinels)

        # compute the baseline, i.e., the expected value of the objective function for a uniform distribution of the
        # budget over all nodes (regardless of the number of sentinels)
        baseline_mean, baseline_stderr = baseline(total_budget=total_budget,
                                                  eval_function=f.evaluate,
                                                  n_nodes=args.n_nodes,
                                                  parallel=args.parallel,
                                                  num_cpu_cores=args.cpu_count)
        list_baseline.append([baseline_mean, baseline_stderr])

        # create a folder to save the results of the individual optimization run
        path_sub_experiment = os.path.join(path_experiment, f'{n}')
        if not os.path.exists(path_sub_experiment):
            os.makedirs(path_sub_experiment)

        # TODO move optimizer calls into a separate utils file
        t0 = time()
        print(f'Optimization {n} start: {strftime("%H:%M:%S", localtime())}\n')
        if args.optimizer == 'cma':

            best_test_strategy, best_solution_history, time_for_optimization = \
                bo_cma(initial_population=prior[0],
                       node_indices=node_indices,
                       n_nodes=args.n_nodes,
                       eval_function=f.evaluate,
                       n_simulations=args.n_simulations,
                       statistic=statistic,
                       total_budget=total_budget,
                       bounds=bounds,
                       path_experiment=path_sub_experiment,
                       max_iterations=args.max_iterations,
                       sigma=args.cma_sigma,
                       parallel=args.parallel,
                       cpu_count=args.cpu_count)

            list_solution_history.append(best_solution_history)
            list_best_test_strategy.append(best_test_strategy)
            list_time_for_optimization.append(time_for_optimization)
            print('------------------------------------------------------')
            print(f'Optimization {n} end: {strftime("%H:%M:%S", localtime())}\n')

            # TODO temporary fix because cma-es does not return stderr
            stderr_history = [best_solution_history, best_solution_history]

        elif args.optimizer == 'gpgo':

            best_test_strategy, best_solution_history, time_for_optimization, time_history, stderr_history =\
                bo_pyGPGO(node_indices=node_indices,
                          n_nodes=args.n_nodes,
                          eval_function=f.evaluate,
                          n_simulations=args.n_simulations,
                          total_budget=total_budget,
                          max_iterations=args.max_iterations,
                          parallel=args.parallel,
                          cpu_count=args.cpu_count,
                          prior=prior,
                          acquisition_function=acquisition_function,
                          use_prior=args.use_prior)

            list_solution_history.append(best_solution_history)
            list_best_test_strategy.append(best_test_strategy)
            list_time_for_optimization.append(time_for_optimization)

            print('------------------------------------------------------')
            print(f'Optimization {n} end: {strftime("%H:%M:%S", localtime())}\n')

            plt.clf()
            plt.plot(range(len(time_history)), time_history[:, 0], label='acquisition function')
            plt.plot(range(len(time_history)), time_history[:, 1], label='surrogate function')
            plt.title('Time for surrogate update and acquisition optimization')
            plt.xlabel('Iteration')
            plt.ylabel('Time in minutes')
            plt.legend()
            plt.savefig(os.path.join(path_sub_experiment, 'gp_and_acqui_time.png'))

        #
        plot_optimizer_history(best_solution_history, stderr_history,
                               baseline_mean, baseline_stderr,
                               args.n_nodes, args.sentinels,
                               path_sub_experiment, args.optimizer)

        plot_time_for_optimization(time_for_optimization,
                                   args.n_nodes, args.sentinels,
                                   path_sub_experiment, args.optimizer)

        best_test_strategy = total_budget * np.exp(best_test_strategy) / sum(np.exp(best_test_strategy))
        best_test_strategy = map_low_dim_x_to_high_dim(best_test_strategy, args.n_nodes, node_indices)

        eval_best_test_strategy, best_test_strategy_stderr = f.evaluate(best_test_strategy,
                                                                        n_simulations=args.n_simulations,
                                                                        parallel=args.parallel,
                                                                        num_cpu_cores=args.cpu_count)

        save_results(best_test_strategy, path_experiment=path_sub_experiment,
                     best_test_strategy_stderr=best_test_strategy_stderr,
                     eval_best_test_strategy=eval_best_test_strategy,
                     baseline_stderr=baseline_stderr, baseline_mean=baseline_mean, t0=t0)



    #TODO save averaged output

    # postprocessing
    print('shape list_solution_history: ', np.shape(list_solution_history))
    print('shape list_best_parameter: ', np.shape(list_best_test_strategy))
    print('shape list_time_for_optimization: ', np.shape(list_time_for_optimization))

    if args.plot_prior:
        y_prior = []
        print(f'Evaluating prior {args.n_runs} times.')
        for _ in tqdm(range(args.n_runs)):
            y_prior.append(evaluate_prior(prior, args.n_simulations, f.evaluate, args.parallel, args.cpu_count))
        y_prior = np.array(y_prior)

        y_prior_mean = np.mean(y_prior[:, :, 0], axis=0)
        y_prior_stderr = np.mean(y_prior[:, :, 1], axis=0)

        # plot the objective function values of the prior
        with open(os.path.join(args.path_plot, f'prior_parameter_{args.n_nodes}_nodes.txt'), 'w') as fi:
            fi.write(prior_parameter)
        plot_prior(prior=prior,
                   path_experiment=args.path_plot,
                   n_nodes=args.n_nodes,
                   y_prior_mean=y_prior_mean,
                   y_prior_stderr=y_prior_stderr,
                   n_runs=args.n_runs)
