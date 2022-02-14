import cma
import matplotlib.pyplot
import os
import numpy as np


# TODO sigma should be about 1/4th of the search space width e.g sigma 30 for budget 120
def bo_cma(objective_function, initial_population, max_iterations, n_simulations, node_indices, n_nodes, eval_function,
           bounds, path_plot, statistic, total_budget, sigma=0.4):
    """
    Runs CMA-ES on the objective function, finding the inputs x for which the output y is minimal.
    @param total_budget: float,
    @param statistic: function object,
    @param path_plot: string,
    @param objective_function: function object,
    @param eval_function: function object,
    @param bounds: list,
    @param n_simulations: int,
    @param node_indices: list,
    @param n_nodes: int,
    @param initial_population: numpy array,
    @param sigma: float,
    @param max_iterations: int,
    @return: list of lists, each list represents an optimal solution
    """
    ea = cma.fmin(objective_function, initial_population, sigma0=sigma,
                  options={'maxiter': max_iterations, 'verbose': -8, 'bounds': bounds},
                  args=(n_simulations, node_indices, n_nodes, eval_function, statistic, total_budget))

    solutions = ea[-2].pop_sorted

    # TODO change x-axis in pot to iterations instead of function evals
    ea[-1].plot()
    cma.s.figsave = matplotlib.pyplot.savefig
    cma.s.figsave(os.path.join(path_plot), dpi=400)

    return [ea[0]]#solutions


# TODO maybe enforce correct types of params ? To prevent floats where ints are expected
def cma_objective_function(x, n_simulations, node_indices, n_nodes, eval_function, statistic, total_budget):
    """
    An optimizeable objective function.
    Maps a lower dimensional x to their corresponding indices in the input vector of the given objective function.
    The input vector x_true is zero at every index except at the indices of x.

    The sum of all values of x_true (or x) is checked to be smaller or equal to the total budget.
    If this constraint is violated the function return 1e10, otherwise the output of the eva function
    (the evaluate function of the SI-model) for n_simulations is returned.

    @param total_budget: float,
    @param statistic: function object,
    @param x: numpy array,
    @param eval_function: function object,
    @param n_simulations: int, number of times the objective function will run a simulation for averaging the output
    @param node_indices: list, indices of x in the higher dimensional x
    @param n_nodes: int, dimension of the input of the objective function
    @return: float, objective function value at x
    """
    assert np.shape(x) == np.shape(node_indices)
    # create a dummy vector to be filled with the values of x at the appropriate indices
    x_true = np.zeros(n_nodes)
    for i, xi in zip(node_indices, x):
        x_true[i] = xi
    if 0 < x_true.sum() <= total_budget:
        return eval_function(x_true, n_simulations=n_simulations, statistic=statistic)
    else:
        # TODO change to numpy.NaN. CMA-ES handles that as explicit rejection of x
        return np.NaN#1e10     # * x.sum(x)


if __name__ == '__main__':
    # Prints out the all hyperparameters for CMA
    for k in cma.CMAOptions():
        print(k, cma.CMAOptions()[k])