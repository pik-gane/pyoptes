import cma
import matplotlib.pyplot
import os
import numpy as np


def bo_cma(objective_function, initial_population,
           max_iterations,
           n_simulations,
           indices,
           true_size_x,
           eval_function,
           bounds,
           path_plot,
           statistic,
           total_budget,
           sigma=0.2):
    """
    Runs CMA-ES on the objective function, finding the inputs x for which the output y is minimal.
    @param total_budget:
    @param statistic:
    @param path_plot: string,
    @param objective_function: function object
    @param eval_function: function object,
    @param bounds: list,
    @param n_simulations: int,
    @param indices: list,
    @param true_size_x: int,
    @param initial_population: numpy array,
    @param sigma: float,
    @param max_iterations: int,
    @return: list of lists, each list represents an optimal solution
    """
    ea = cma.fmin(objective_function, initial_population, sigma0=sigma,
                  options={'maxiter': max_iterations, 'verbose': -8, 'bounds': bounds},
                  args=(n_simulations, indices, true_size_x, eval_function, statistic, total_budget))

    solutions = ea[-2].pop_sorted

    ea[-1].plot()
    cma.s.figsave = matplotlib.pyplot.savefig
    cma.s.figsave(os.path.join(path_plot), dpi=400)

    return solutions


# TODO find descriptive name for function
# TODO rename true_size_x to n_nodes ??
def cma_objective_function(x, n_simulations, indices, true_size_x, eval_function, statistic, total_budget):
    """
    Maps a lower dimensional x to their corresponding indices in the input vector of the given objective function.
    The input vector x_true is zero at every index except at the indices of x.

    The sum of all values of x_true (or x) is checked to be smaller or equal to the total budget.
    If this constraint is violated the function return 1e10, otherwise the output of the eva function
    (the evaluate function of the SI-model) for n_simulations is returned.

    @type total_budget: object
    @param statistic:
    @param x: numpy array,
    @param eval_function: function object,
    @param n_simulations: int, number of times the objective function will run a simulation for averaging the output
    @param indices: list, indices of x in the higher dimensional x
    @param true_size_x: int, dimension of the input of the objective function
    @return: float, objective function value at x
    """
    assert np.shape(x) == np.shape(indices)
    # create a dummy vector to be filled with the values of x at the appropriate indices
    x_true = np.zeros(true_size_x)
    for i, xi in zip(indices, x):
        x_true[i] = xi
    if x_true.sum() <= total_budget:
        return eval_function(x_true, n_simulations=n_simulations, statistic=statistic)
    else:
        # TODO change to numpy.NaN. CMA-ES handles that as explicit rejection of x
        return 1e10     # * x.sum(x)


if __name__ == '__main__':
    # Prints out the all hyperparameters for CMA
    for k in cma.CMAOptions():
        print(k, cma.CMAOptions()[k])