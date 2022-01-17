import numpy as np
# from scipy.stats import gaussian_kde as kde
# import pylab as plt
from pyoptes import set_seed
from pyoptes.optimization.budget_allocation import target_function as f
import cma
from tqdm import tqdm


def get_mean_function_output(x, function=f.evaluate, n=100):
    """
    Rerun a function n times for the same input x and return the mean of the results
    @param x: valid input for the function
    @param function: any python function
    @param n: integer, number of times the function should run
    @return: the mean of the output value for x
    """
    y_sum = []
    for _ in range(n):
        y_sum.append(function(x))

    y_mean = np.mean(y_sum)
    return y_mean


if __name__ == '__main__':
    # set some seed to get reproducible results:
    set_seed(1)

    print("Preparing the target function for a random but fixed transmissions network")

    # at the beginning, call prepare() once:
    f.prepare()
    n_inputs = f.get_n_inputs()
    print("n_inputs (=number of network nodes):", n_inputs)

    total_budget = 1.0 * n_inputs  # i.e., on average, nodes will do one test per year

    # evaluate f once at a random input:
    weights = np.random.rand(n_inputs)
    shares = weights / weights.sum()
    x = shares * total_budget

    # for _ in tqdm(range(10)):
    #     print(get_mean_function_output(x, n=100))

    # Prints out the all hyperparameters for CMA
    # for k in cma.CMAOptions():
    #     print(k, cma.CMAOptions()[k])

    sigma = 0.2
    max_iterations = 1000
    ea = cma.CMAEvolutionStrategy(x, sigma, inopts={'maxiter': max_iterations, 'verbose': -8})
    ea.optimize(f.evaluate)
    solutions = ea.pop_sorted

    for s in solutions:
        print(get_mean_function_output(s))
