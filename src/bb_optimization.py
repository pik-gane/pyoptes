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
    args = parser.parse_args()
    # set some seed to get reproducible results:
    set_seed(1)

    n_nodes = 120
    n_simulations = 1000
    # at the beginning, call prepare() once:
    f.prepare(n_nodes=n_nodes)
    n_inputs = f.get_n_inputs()

    total_budget = 1.0 * n_inputs  # i.e., on average, nodes will do one test per year

    # evaluate f once at a random input:
    weights = np.random.rand(n_inputs)
    shares = weights / weights.sum()
    x = shares * total_budget

    # weird hack, cma-es only takes function objects so the default value n_simulations of f.evaluate
    # can't be changed. The wrapper "evaluate" fixes only the n_simulations and passes the x to the "real" function
    def objective_function_cma(x, n_simulations=n_simulations):
        return f.evaluate(x, n_simulations=n_simulations)

    def objective_function_alebo(x):#, nn_simulations=nn_simulations):
        x = np.array(list(x.values()))
        # TODO add check whether x satfisfies constraint x0 + .. xi <= B
        return f.evaluate(x)#, n_simulations=nn_simulations), 0.0)}

    if args.optimizer == 'cma':
        solutions = bo_cma(objective_function_cma, x)
    elif args.optimizer == 'alebo':
        r = bo_alebo(objective_function_alebo, n_nodes)

    else:
        print('Something went wrong with choosing the optimizer.')
