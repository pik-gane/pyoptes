import numpy as np
from pyoptes import set_seed
from pyoptes.optimization.budget_allocation import target_function as f
import cma
from tqdm import tqdm


if __name__ == '__main__':
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

    # for _ in tqdm(range(10)):
    #     print(get_mean_function_output(x, n=100))

    # # Prints out the all hyperparameters for CMA
    # for k in cma.CMAOptions():
    #     print(k, cma.CMAOptions()[k])

    # weird hack, cma-es only takes function objects so the default value n_simulations of f.evaluate
    # can't be changed. The wrapper "evaluate" fixes only the n_simulations and passes the x to the "real" function
    def evaluate(x, n_simulations=n_simulations):
        return f.evaluate(x, n_simulations=n_simulations)

    sigma = 0.2
    max_iterations = 1000
    ea = cma.CMAEvolutionStrategy(x, sigma, inopts={'maxiter': max_iterations, 'verbose': -8})
    ea.optimize(evaluate)
    solutions = ea.pop_sorted

    for s in solutions:
        print(f.evaluate(s, n_simulations=10000))
