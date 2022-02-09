from pyoptes import set_seed
from pyoptes.optimization.budget_allocation import target_function as f

from pyoptes.optimization.budget_allocation.blackbox_learning.bo_cma import bo_cma, cma_objective_function
from pyoptes.optimization.budget_allocation.blackbox_learning.bo_alebo import bo_alebo
from pyoptes.optimization.budget_allocation.blackbox_learning.utils import choose_high_degree_nodes

import numpy as np
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("optimizer", choices=['cma', 'alebo'],
                        help="Choose the optimizer to run on the SI-model. Choose between CMA-ES and ALEBO")
    parser.add_argument("--n_nodes", type=int, default=120, help="Defines the number of nodes used by the SI-model. "
                                                                 "Default value is 120.")
    parser.add_argument("--n_simulations", type=int, default=1000, help=" Sets the number of runs the for the "
                                                                        "SI-model. Higher values of n_simulations lower"
                                                                        "the variance of the output of the simulation. "
                                                                        "Default value is 1000.")
    parser.add_argument("--max_iterations", type=int, default=100, help="The maximum number of iterations CMA-ES runs.")
    parser.add_argument("--size_subset", type=int, default=10, help="Set the number of nodes that are used. Has to be"
                                                                    "smaller than or equal to n_nodes")
    parser.add_argument('--cma_sigma', type=float, default=0.2, help="")
    parser.add_argument('--path_plot', default='pyoptes/optimization/budget_allocation/blackbox_learning/plots/cma-es',
                        help="location and name of the file a plot of the results of CMA-ES is saved to.")
    parser.add_argument('--solution_initialisation', choices=['uniform', 'random'], default='random',
                        help="")
    args = parser.parse_args()

    # set some seed to get reproducible results:
    set_seed(1)
    # at the beginning, call prepare() once:
    f.prepare(n_nodes=args.n_nodes)

    total_budget = 1.0 * args.n_nodes  # i.e., on average, nodes will do one test per year

    if args.solution_initialisation == 'random':
        # evaluate f once at a random input:
        weights = np.random.rand(args.size_subset)
        shares = weights / weights.sum()
        x = shares * total_budget
    # TODO maybe change to np.random.uniform ??
    elif args.solution_initialisation == 'uniform':
        # distribute total budget uniformly
        x = np.array([total_budget/args.size_subset for _ in range(args.size_subset)])
    # TODO exponential distribution ?
    else:
        raise Exception('Invalid solution initialisation chosen.')

    # reduce the dimension of the input space
    ix = choose_high_degree_nodes(f.network.degree, args.size_subset)

    # get the first constraint, the boundaries of x_i
    bounds = [0, total_budget]

    # define function to average the results of the simulation
    # the square of the mean is taken to emphasise the tail of the distribution of y
    statistic = lambda x: np.mean(x)**2

    # def objective_function_alebo(x, total_budget=total_budget):#, nn_simulations=nn_simulations):
    #     x = np.array(list(x.values()))
    #     if x.sum() <= 120.0:
    #         return f.evaluate(x)#, n_simulations=nn_simulations), 0.0)}
    #     else:
    #         return 1e10     # * x.sum(x)

    # print('transmission array', f.model.transmissions_array)

    # print('capacities per node: ', f.capacities)

    if args.optimizer == 'cma':
        solutions = bo_cma(cma_objective_function, x,
                           indices=ix,
                           true_size_x=args.n_nodes,
                           eval_function=f.evaluate,
                           n_simulations=args.n_simulations,
                           statistic=statistic,
                           total_budget=total_budget,
                           bounds=bounds,
                           path_plot=args.path_plot,
                           max_iterations=args.max_iterations)
        print(f'\nBest CMA-ES solutions evaluated on {args.n_simulations} simulations, descending ')
        for s in solutions:
            print(cma_objective_function(s,
                                         indices=ix,
                                         true_size_x=args.n_nodes,
                                         eval_function=f.evaluate,
                                         n_simulations=args.n_simulations,
                                         statistic=statistic,
                                         total_budget=total_budget))
    elif args.optimizer == 'alebo':
        best_parameters, values, experiment, model = bo_alebo(objective_function_alebo, args.n_nodes,
                                                              args.max_iterations)

        best_parameters = np.array(list(best_parameters.values()))
        print('min, max, sum: ', best_parameters.min(), best_parameters.max(), best_parameters.sum())

    else:
        print('Something went wrong with choosing the optimizer.')
