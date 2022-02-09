import cma
import matplotlib.pyplot
import os


def bo_cma(objective_function, initial_population,
           max_iterations,
           n_simulations,
           indices,
           true_size_x,
           eval_function,
           bounds,
           path_plot,
           statistic,
           sigma=0.2):
    """
    Runs CMA-ES on the objective function, finding the inputs x for which the output y is minimal.
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
                  args=(n_simulations, indices, true_size_x, eval_function, statistic))

    solutions = ea[-2].pop_sorted

    ea[-1].plot()
    cma.s.figsave = matplotlib.pyplot.savefig
    cma.s.figsave(os.path.join(path_plot), dpi=400)

    return solutions


if __name__ == '__main__':
    # Prints out the all hyperparameters for CMA
    for k in cma.CMAOptions():
        print(k, cma.CMAOptions()[k])