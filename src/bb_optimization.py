import os.path

from pyoptes import set_seed
from pyoptes.optimization.budget_allocation import target_function as f

from pyoptes import bo_smac
from pyoptes import bo_alebo
from pyoptes import bo_cma, cma_objective_function
from pyoptes.optimization.budget_allocation.blackbox_learning.bo_pyGPGO import bo_pyGPGO

from pyoptes import choose_high_degree_nodes, baseline
from pyoptes import map_low_dim_x_to_high_dim, test_function, create_test_strategy_prior
from pyoptes import save_hyperparameters, save_results

import inspect
import argparse
import numpy as np
import networkx as nx
from scipy.stats import lognorm
from time import time, localtime, strftime


#
def caps(size):
    return lognorm.rvs(s=2, scale=np.exp(4), size=size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("optimizer", choices=['cma', 'alebo', 'smac', 'gpgo'],
                        help="Choose the optimizer to run on the SI-model. Choose between CMA-ES and ALEBO")
    parser.add_argument("name_experiment", help="The name of the folder where the results of the optimizer run are"
                                                "saved to.")

    parser.add_argument("--sentinels", type=int, default=10,
                        help="Set the number of nodes that are used. Has to be smaller than or equal to n_nodes. "
                             "Default is 10 nodes.")
    parser.add_argument("--n_nodes", type=int, default=120,
                        help="Si-simulation parameter. "
                             "Defines the number of nodes used by the SI-model to create a graph. "
                             "Default value is 120 nodes.")

    parser.add_argument('--test_strategy_initialisation', choices=['uniform', 'random'], default='uniform',
                        help="Defines how the initial test strategy is initialised.")
    parser.add_argument("--n_simulations", type=int, default=1000,
                        help="Si-simulation parameter. Sets the number of runs the for the SI-model. "
                             "Higher values of n_simulations lower the variance of the output of the simulation. "
                             "Default value is 1000.")
    # TODO rename graph to something more clear ??
    parser.add_argument('--graph', choices=['waxman', 'barabasi-albert'], default='barabasi-albert',
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
    parser.add_argument("--cpu_count", type=int, default=12,
                        help='Si-simulation parameter. Defines the number of cpus to be used for the simulation '
                             'parallelization. If more cpus are chosen than available, the max available are selected.'
                             '-1 selects all available cpus. Default are 12 cpus.')

    parser.add_argument("--max_iterations", type=int, default=1000,
                        help="Optimizer parameter. The maximum number of iterations the algorithms run.")
    parser.add_argument('--cma_sigma', type=float, default=30,
                        help="Optimizer parameter. Defines the variance in objective function parameters "
                             "from which new population is sampled. Therefore the variance has to be big enough to"
                             "change the parameters in a meaningful way. A useful heuristic is to set the variance to "
                             "about 1/4th of the parameter search space. Default value (for 120 nodes) is 30.")
    parser.add_argument('--acquisition_function', default='ExpectedImprovement', choices=['ExpectedImprovement'],
                        help='GPGO optimizer parameter. Defines the acquisition function that is used by GPGO.')
    parser.add_argument('--use_prior', type=bool, default=True,
                        help='')

    parser.add_argument("--log_level", type=int, default=3, choices=range(1, 11), metavar="[1-10]",
                        help="Optimizer parameter. Only effects SMAC and GPGO. Sets how often log messages appear. "
                             "Lower values mean more messages.")
    parser.add_argument('--path_plot', default='pyoptes/optimization/budget_allocation/blackbox_learning/plots/',
                        help="Optimizer parameter. Location where all the individual results"
                             " of the optimizers are saved to. "
                             "Default location is 'pyoptes/optimization/budget_allocation/blackbox_learning/plots/'")
    args = parser.parse_args()

    if args.graph == 'waxman':
        # generate a Waxman graph:
        waxman = nx.waxman_graph(120)
        pos = dict(waxman.nodes.data('pos'))
        # convert into a directed graph:
        static_network = nx.DiGraph(nx.to_numpy_array(waxman))
    elif args.graph == 'barabasi-albert':
        static_network = None
    else:
        raise Exception(f'"{args.graph}" is an invalid choice for the graph.')

    # sets seed for SI-simulation
    set_seed(1)
    f.prepare(n_nodes=args.n_nodes,
              capacity_distribution=caps,
              p_infection_by_transmission=args.p_infection_by_transmission,
              static_network=static_network,
              delta_t_symptoms=args.delta_t_symptoms)

    total_budget = 1.0 * args.n_nodes  # i.e., on average, nodes will do one test per year

    #
    if args.test_strategy_initialisation == 'random':
        weights = np.random.rand(args.sentinels)
        shares = weights / weights.sum()
        initial_x = shares * total_budget
    elif args.test_strategy_initialisation == 'uniform':
        # distribute total budget uniformly
        initial_x = np.array([total_budget/args.sentinels for _ in range(args.sentinels)])
    else:
        raise Exception('Invalid test_strategy initialisation chosen.')

    # reduce the dimension of the input space by choosing to only allocate the budget between nodes with the highest
    # degrees. The function return the indices of these nodes
    node_indices = choose_high_degree_nodes(f.network.degree, args.sentinels)

    # get the first constraint, the boundaries of x_i
    bounds = [0, total_budget]

    # define function to average the results of the simulation
    # the mean of the squared ys is taken to emphasise the tail of the distribution of y
    statistic = lambda x: np.mean(x**2, axis=0)

    prior = create_test_strategy_prior(args.n_nodes, f.network.degree, f.capacities, total_budget)

    # print(np.shape(prior))
    # print([x.sum() for x in prior])

    # compute the baseline value for y
    baseline = baseline(initial_x,
                        eval_function=f.evaluate,
                        n_nodes=args.n_nodes,
                        node_indices=node_indices,
                        statistic=statistic,
                        parallel=args.parallel,
                        num_cpu_cores=args.cpu_count
                        )

    # save SI-model parameters as .json-file
    experiment_params = {'simulation_hyperparameters': {'node_initialisation': args.test_strategy_initialisation,
                                                        'total_budget': total_budget,
                                                        'n_nodes': args.n_nodes,
                                                        'graph': args.graph,
                                                        'sentinels': args.sentinels,
                                                        'n_simulations': args.n_simulations,
                                                        'statistic': inspect.getsourcelines(statistic)[0][0][23:-1],
                                                        'delta_t_symptoms': args.delta_t_symptoms,
                                                        'p_infection_by_transmission': args.p_infection_by_transmission
                                                        },
                         'optimizer_hyperparameters': {'optimizer': args.optimizer,
                                                       'max_iterations': args.max_iterations,
                                                       }}

    if args.optimizer == 'cma':
        experiment_params['optimizer_hyperparameters']['cma_sigma'] = args.cma_sigma
        path_experiment = os.path.join(args.path_plot, args.name_experiment)
        save_hyperparameters(experiment_params, path_experiment)
        print('saved hyperparameters')
        print(f'Optimization start: {strftime("%H:%M:%S", localtime())}')
        t0 = time()
        best_parameter = bo_cma(objective_function=cma_objective_function,
                                initial_population=initial_x,
                                node_indices=node_indices,
                                n_nodes=args.n_nodes,
                                eval_function=f.evaluate,
                                n_simulations=args.n_simulations,
                                statistic=statistic,
                                total_budget=total_budget,
                                bounds=bounds,
                                path_experiment=path_experiment,
                                max_iterations=args.max_iterations,
                                sigma=args.cma_sigma,
                                parallel=args.parallel,
                                cpu_count=args.cpu_count)

        print('------------------------------------------------------')
        print(f'Optimization end: {strftime("%H:%M:%S", localtime())}')

        eval_best_parameter = cma_objective_function(best_parameter,
                                                     node_indices=node_indices,
                                                     n_nodes=args.n_nodes,
                                                     eval_function=f.evaluate,
                                                     n_simulations=args.n_simulations,
                                                     statistic=statistic,
                                                     total_budget=total_budget,
                                                     parallel=args.parallel,
                                                     cpu_count=args.cpu_count)
        p = f'\nParameters:\nSentinel nodes: {args.sentinels}\nn_nodes: {args.n_nodes}' \
            f'\niterations: {args.max_iterations}\nn_simulations: {args.n_simulations}' \
            f'\nTime for optimization (in minutes): {(time() - t0) / 60}' \
            f'\n\nBaseline for {args.test_strategy_initialisation} budget distribution: {baseline[str(args.n_simulations)]}' \
            f'\nBest CMA-ES solutions:' \
            f'\nObjective value:  {eval_best_parameter}' \
            f'\nx min, x max, x sum: {best_parameter.min()}, {best_parameter.max()}, {best_parameter.sum()}'

        save_results(best_parameter, eval_output=p, path_experiment=path_experiment)
        print(p)

    elif args.optimizer == 'alebo':
        path_experiment = os.path.join(args.path_plot, args.name_experiment)
        save_hyperparameters(experiment_params, path_experiment)

        t0 = time()
        best_parameters, values, experiment, model = bo_alebo(n_nodes=args.n_nodes,
                                                              total_trials=args.max_iterations,
                                                              indices=node_indices,
                                                              eval_function=f.evaluate,
                                                              n_simulations=args.n_simulations,
                                                              statistic=statistic,
                                                              total_budget=total_budget,
                                                              path_plot=args.path_plot)
        print(f'Optimization end: {strftime("%H:%M:%S", localtime())}')

        print(f'\nParameters:\nSentinel nodes: {args.sentinels}\nn_nodes: {args.n_nodes}\n'
              f'n_simulations: {args.n_simulations}\nNetwork type: {args.graph}')
        print(f'Baseline for {args.test_strategy_initialisation} budget distribution: {baseline[str(args.n_simulations)]}')
        best_parameters = np.array(list(best_parameters.values()))
        print('min, max, sum: ', best_parameters.min(), best_parameters.max(), best_parameters.sum())

    elif args.optimizer == 'smac':
        path_experiment = os.path.join(args.path_plot, args.name_experiment)
        save_hyperparameters(experiment_params, path_experiment)
        print('saved hyperparameters')
        print(f'Optimization start: {strftime("%H:%M:%S", localtime())}')
        t0 = time()
        best_parameter = bo_smac(initial_population=initial_x,
                                 node_indices=node_indices,
                                 n_nodes=args.n_nodes,
                                 eval_function=test_function,
                                 n_simulations=args.n_simulations,
                                 statistic=statistic,
                                 total_budget=total_budget,
                                 max_iterations=args.max_iterations,
                                 path_experiment=path_experiment,
                                 parallel=args.parallel,
                                 cpu_count=args.cpu_count,
                                 log_level=args.log_level)
        print(f'Optimization end: {strftime("%H:%M:%S", localtime())}')

        best_parameter = np.array(list(best_parameter.values()))
        best_parameter = map_low_dim_x_to_high_dim(best_parameter, args.n_nodes, node_indices)
        best_parameter = best_parameter/best_parameter.sum()
        best_parameter = best_parameter*total_budget

        eval_best_parameter = f.evaluate(best_parameter,
                                         n_simulations=args.n_simulations,
                                         statistic=statistic,
                                         parallel=args.parallel,
                                         num_cpu_cores=args.cpu_count)

        p = f'\nParameters:\nSentinel nodes: {args.sentinels}\nn_nodes: {args.n_nodes}' \
            f'\niterations: {args.max_iterations}\nn_simulations: {args.n_simulations}' \
            f'\nTime for optimization (in minutes): {(time() - t0) / 60}' \
            f'\n\nBaseline for {args.test_strategy_initialisation} budget distribution: {baseline[str(args.n_simulations)]}' \
            f'\nBest SMAC solutions:' \
            f'\nObjective value:  {eval_best_parameter}' \
            f'\nx min, x max, x sum: {best_parameter.min()}, {best_parameter.max()}, {best_parameter.sum()}'

        save_results(best_parameter, eval_output=p, path_experiment=path_experiment)
        print(p)

    elif args.optimizer == 'gpgo':
        experiment_params['optimizer_hyperparameters']['use_prior'] = args.use_prior
        experiment_params['optimizer_hyperparameters']['acquisition_function'] = args.acquisition_function
        path_experiment = os.path.join(args.path_plot, args.name_experiment)
        save_hyperparameters(experiment_params, path_experiment)
        print('saved hyperparameters')
        print(f'Optimization start: {strftime("%H:%M:%S", localtime())}')
        t0 = time()

        best_parameter = bo_pyGPGO(node_indices=node_indices,
                                   n_nodes=args.n_nodes,
                                   eval_function=f.evaluate,
                                   n_simulations=args.n_simulations,
                                   statistic=statistic,
                                   total_budget=total_budget,
                                   path_experiment=path_experiment,
                                   max_iterations=args.max_iterations,
                                   parallel=args.parallel,
                                   cpu_count=args.cpu_count,
                                   log_level=args.log_level,
                                   prior=prior,
                                   acquisition_function=args.acquisition_function,
                                   use_prior=args.use_prior)
        print('------------------------------------------------------')
        print(f'Optimization end: {strftime("%H:%M:%S", localtime())}')

        best_parameter = list(best_parameter[0].values())
        best_parameter = total_budget * np.exp(best_parameter) / sum(np.exp(best_parameter))
        best_parameter = map_low_dim_x_to_high_dim(best_parameter, args.n_nodes, node_indices)

        eval_best_parameter = f.evaluate(best_parameter, n_simulations=args.n_simulations, statistic=statistic,
                                         parallel=args.parallel, num_cpu_cores=args.cpu_count)

        p = f'\nParameters:\nSentinel nodes: {args.sentinels}\nn_nodes: {args.n_nodes}' \
            f'\niterations: {args.max_iterations}\nn_simulations: {args.n_simulations}' \
            f'\nTime for optimization (in hours): {(time() - t0) / 3600}' \
            f'\n\nBaseline for {args.test_strategy_initialisation} budget distribution: {baseline[str(args.n_simulations)]}' \
            f'\nBest GPGO solutions:' \
            f'\nObjective value:  {eval_best_parameter}' \
            f'\nx min, x max, x sum: {best_parameter.min()}, {best_parameter.max()}, {best_parameter.sum()}'

        save_results(best_parameter, eval_output=p, path_experiment=path_experiment)
        print(p)

    else:
        print('Something went wrong with choosing the optimizer.')
