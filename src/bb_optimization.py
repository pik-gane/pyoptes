import os.path

from pyoptes import set_seed
from pyoptes.optimization.budget_allocation import target_function as f

from pyoptes.optimization.budget_allocation.blackbox_learning.bo_cma import bo_cma, cma_objective_function
from pyoptes.optimization.budget_allocation.blackbox_learning.bo_alebo import bo_alebo
from pyoptes.optimization.budget_allocation.blackbox_learning.bo_smac import bo_smac

from pyoptes.optimization.budget_allocation.blackbox_learning.utils import choose_high_degree_nodes, baseline
from pyoptes.optimization.budget_allocation.blackbox_learning.utils import map_low_dim_x_to_high_dim, test_function
from pyoptes.optimization.budget_allocation.blackbox_learning.utils import save_parameters

import json
import inspect
import argparse
import numpy as np
import networkx as nx


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("optimizer", choices=['cma', 'alebo', 'smac'],
                        help="Choose the optimizer to run on the SI-model. Choose between CMA-ES and ALEBO")
    parser.add_argument("name_experiment", help="")
    parser.add_argument('--path_plot', default='pyoptes/optimization/budget_allocation/blackbox_learning/plots/',
                        help="location the results of the optimizers are saved to.")
    parser.add_argument("--n_nodes", type=int, default=120,
                        help="Defines the number of nodes used by the SI-model. Default value is 120.")
    parser.add_argument("--n_simulations", type=int, default=1000,
                        help=" Sets the number of runs the for the SI-model. Higher values of n_simulations lower "
                             "the variance of the output of the simulation. Default value is 1000.")
    parser.add_argument("--max_iterations", type=int, default=100,
                        help="The maximum number of iterations the algorithms run.")
    # TODO rename size_subset to sentinels ?
    parser.add_argument("--sentinels", type=int, default=10,
                        help="Set the number of nodes that are used. Has to be smaller than or equal to n_nodes")
    parser.add_argument('--cma_sigma', type=float, default=30,
                        help="Defines the variance in objective function parameters from which new population is sampled")
    parser.add_argument('--solution_initialisation', choices=['uniform', 'random'], default='uniform',
                        help="")
    # TODO rename to something more clear + clear up help
    parser.add_argument('--graph', choices=['waxman', 'barabasi-albert'], default='barabasi-albert',
                        help='Set the type of graph the ... . Either Waxman or Barabasi-Albert (ba) can be used')
    args = parser.parse_args()

    # TODO add support for reading parameters from json-files // or save parameters + results in a json
    # set some seed to get reproducible results:
    set_seed(1)

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

    # at the beginning, call prepare() once:
    f.prepare(n_nodes=args.n_nodes, capacity_distribution=np.random.lognormal,
              p_infection_by_transmission=0.5, static_network=static_network, delta_t_symptoms=60)

    total_budget = 1.0 * args.n_nodes  # i.e., on average, nodes will do one test per year

    if args.solution_initialisation == 'random':
        # evaluate f once at a random input:
        weights = np.random.rand(args.sentinels)
        shares = weights / weights.sum()
        x = shares * total_budget
    elif args.solution_initialisation == 'uniform':
        # distribute total budget uniformly
        x = np.array([total_budget/args.sentinels for _ in range(args.sentinels)])
    # TODO exponential distribution ?
    else:
        raise Exception('Invalid solution initialisation chosen.')

    # reduce the dimension of the input space
    node_indices = choose_high_degree_nodes(f.network.degree, args.sentinels)

    # get the first constraint, the boundaries of x_i
    bounds = [0, total_budget]

    # define function to average the results of the simulation
    # the mean of the squared ys is taken to emphasise the tail of the distribution of y
    statistic = lambda x: np.mean(x**2)

    # compute the baseline value for y
    baseline = baseline(x, eval_function=f.evaluate, n_nodes=args.n_nodes, node_indices=node_indices, statistic=statistic)

    # save SI-model parameters as .json-file
    experiment_params = {'model': {'node_initialisation': args.solution_initialisation,
                                   'total_budget': total_budget,
                                   'n_nodes': args.n_nodes,
                                   'graph': args.graph,
                                   'sentinels': args.sentinels,
                                   'n_simulations': args.n_simulations,
                                   'max_iterations': args.max_iterations,
                                   'statistic': inspect.getsourcelines(statistic)[0][0][23:-1],
                                   },
                         'optimizer': {}}

    # # TODO log the beste and baseline results somewhere
    if args.optimizer == 'cma':
        experiment_params['optimizer']['cma_sigma'] = args.cma_sigma
        path_experiment = os.path.join(args.path_plot, args.name_experiment)
        save_parameters(experiment_params, path_experiment)

        solutions = bo_cma(objective_function=cma_objective_function,
                           initial_population=x,
                           node_indices=node_indices,
                           n_nodes=args.n_nodes,
                           eval_function=f.evaluate,
                           n_simulations=args.n_simulations,
                           statistic=statistic,
                           total_budget=total_budget,
                           bounds=bounds,
                           path_experiment=path_experiment,
                           max_iterations=args.max_iterations,
                           sigma=args.cma_sigma)
        print(f'Parameters:\nSentinel nodes: {args.sentinels}\nn_nodes: {args.n_nodes}\nn_simulations: {args.n_simulations}')
        print(f'Baseline for {args.solution_initialisation} budget distribution: {baseline[str(args.n_simulations)]}')
        print(f'\nBest CMA-ES solutions:')

        for s in solutions:
            print(f'x sum: {s.sum()} || Objective value: ', cma_objective_function(s,
                                                                                   node_indices=node_indices,
                                                                                   n_nodes=args.n_nodes,
                                                                                   eval_function=f.evaluate,
                                                                                   n_simulations=args.n_simulations,
                                                                                   statistic=statistic,
                                                                                   total_budget=total_budget),
                  f'\n\tmin, max: {s.min()}, {s.max()}')#,
                  # f'\n\t{s}')
    elif args.optimizer == 'alebo':
        best_parameters, values, experiment, model = bo_alebo(n_nodes=args.n_nodes,
                                                              total_trials=args.max_iterations,
                                                              indices=node_indices,
                                                              eval_function=f.evaluate,
                                                              n_simulations=args.n_simulations,
                                                              statistic=statistic,
                                                              total_budget=total_budget,
                                                              path_plot=args.path_plot)

        print(f'Parameters:\nSentinel nodes: {args.sentinels}\nn_nodes: {args.n_nodes}\n'
              f'n_simulations: {args.n_simulations}\nNetwork type: {args.graph}')
        print(f'Baseline for {args.solution_initialisation} budget distribution: {baseline[str(args.n_simulations)]}')
        best_parameters = np.array(list(best_parameters.values()))
        print('min, max, sum: ', best_parameters.min(), best_parameters.max(), best_parameters.sum())

    elif args.optimizer == 'smac':

        best_parameters = bo_smac(initial_population=x,
                                  node_indices=node_indices,
                                  n_nodes=args.n_nodes,
                                  eval_function=f.evaluate,
                                  n_simulations=args.n_simulations,
                                  statistic=statistic,
                                  total_budget=total_budget,
                                  max_iterations=args.max_iterations)

        print(f'Parameters:\nSentinel nodes: {args.sentinels}\nn_nodes: {args.n_nodes}\nn_simulations: {args.n_simulations}')
        print(f'Baseline for {args.solution_initialisation} budget distribution: {baseline[str(args.n_simulations)]}')

        best_parameters = np.array(list(best_parameters.values()))
        best_parameters = map_low_dim_x_to_high_dim(best_parameters, args.n_nodes, node_indices)
        print('min, max, sum: ', best_parameters.min(), best_parameters.max(), best_parameters.sum())
        print(f'Objective value: {f.evaluate(best_parameters, n_simulations=args.n_simulations, statistic=statistic)}')

    else:
        print('Something went wrong with choosing the optimizer.')
