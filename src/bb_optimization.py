from pyoptes import set_seed
from pyoptes.optimization.budget_allocation import target_function as f

from pyoptes.optimization.budget_allocation.blackbox_learning.bo_cma import bo_cma
from pyoptes.optimization.budget_allocation.blackbox_learning.bo_alebo import bo_alebo

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
    parser.add_argument("--max_iterations", type=int, default=100)
    args = parser.parse_args()

    # set some seed to get reproducible results:
    set_seed(1)
    # at the beginning, call prepare() once:
    f.prepare(n_nodes=args.n_nodes)

    total_budget = 1.0 * args.n_nodes  # i.e., on average, nodes will do one test per year
    # evaluate f once at a random input:
    weights = np.random.rand(args.n_nodes)
    shares = weights / weights.sum()
    x = shares * total_budget

    # weird hack, cma-es only takes function objects so the default value n_simulations of f.evaluate
    # can't be changed. The wrapper "evaluate" fixes only the n_simulations and passes the x to the "real" function
    def objective_function_cma(x, n_simulations=args.n_simulations):
        if x.sum() <= 120.0:
            return f.evaluate(x, n_simulations=n_simulations)
        else:
            return 1e10     # * x.sum(x)

    def objective_function_alebo(x, total_budget=total_budget):#, nn_simulations=nn_simulations):
        x = np.array(list(x.values()))
        # TODO add check whether x satisfies constraint x0 + .. xi <= B
        # print(type(x.sum()), x.sum())
        # print(type(total_budget), total_budget)
        if x.sum() <= 120.0:
            return f.evaluate(x)#, n_simulations=nn_simulations), 0.0)}
        else:
            return 1e10     # * x.sum(x)

    if args.optimizer == 'cma':
        solutions = bo_cma(objective_function_cma, x, max_iterations=args.max_iterations)
        # print(type(x.sum()), x.sum())
        # for s in solutions:
        #     print(solutions)
    elif args.optimizer == 'alebo':
        best_parameters, values, experiment, model = bo_alebo(objective_function_alebo, args.n_nodes,
                                                              args.max_iterations)

        best_parameters = np.array(list(best_parameters.values()))
        print('min, max, sum: ', best_parameters.min(), best_parameters.max(), best_parameters.sum())

    else:
        print('Something went wrong with choosing the optimizer.')
