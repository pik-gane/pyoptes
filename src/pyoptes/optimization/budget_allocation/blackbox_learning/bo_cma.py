import cma
import matplotlib.pyplot
import os
import numpy as np

from .utils import map_low_dim_x_to_high_dim


# TODO complete documentation for parameters
# TODO sigma should be about 1/4th of the search space width e.g sigma 30 for budget 120
def bo_cma(objective_function, initial_population, max_iterations, n_simulations, node_indices, n_nodes, eval_function,
           bounds, path_experiment, statistic, total_budget, parallel, cpu_count, sigma=0.4):
    """
    Runs CMA-ES on the objective function, finding the inputs x for which the output y is minimal.
    @param total_budget: float, the total budget that is to be distributed along the nodes of the graph
    @param statistic: function object,
    @param path_experiment: string,
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
                  args=(n_simulations, node_indices, n_nodes, eval_function,
                        statistic, total_budget, parallel, cpu_count))

    solutions = ea[-2].pop_sorted   # pop_sorted is the population after stopping CMA ||

    # TODO change x-axis in plot to iterations instead of function evals
    ea[-1].plot()
    cma.s.figsave = matplotlib.pyplot.savefig
    path_plot = os.path.join(path_experiment, 'cma_plot.png')
    cma.s.figsave(path_plot, dpi=400)

    return ea[0] #solutions    # contains the best solution found during the whole run


# TODO maybe enforce correct types of params ? To prevent floats where ints are expected
def cma_objective_function(x, n_simulations, node_indices, n_nodes, eval_function,
                           statistic, total_budget, parallel, cpu_count):
    """
    An optimizeable objective function.
    Maps a lower dimensional x to their corresponding indices in the input vector of the given objective function.
    The input vector x_true is zero at every index except at the indices of x.

    The sum of all values of x_true (or x) is checked to be smaller or equal to the total budget.
    If this constraint is violated the function return 1e10, otherwise the output of the eva function
    (the evaluate function of the SI-model) for n_simulations is returned.

    @param cpu_count:
    @param parallel:
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

    x = map_low_dim_x_to_high_dim(x, n_nodes, node_indices)

    if 0 < x.sum() <= total_budget:
        return eval_function(x, n_simulations=n_simulations, statistic=statistic,
                             parallel=parallel, num_cpu_cores=cpu_count)
    else:
        return np.NaN#1e10     # * x.sum(x)


if __name__ == '__main__':
    # Prints out the all hyperparameters for CMA
    for k in cma.CMAOptions():
        print(k, cma.CMAOptions()[k])