import os
import cma
import time
import numpy as np
from tqdm import tqdm

from .utils import bo_map_low_dim_x_to_high_dim, bo_softmax


# TODO complete documentation for parameters
def bo_cma(initial_population, max_iterations, n_simulations, node_indices, n_nodes, eval_function,
           bounds, statistic, total_budget, parallel, num_cpu_cores, sigma, popsize,
           save_test_strategies=False, save_test_strategies_path=None):
    """
    Runs CMA-ES on the objective function, finding the inputs x for which the output y is minimal.
    @param popsize: int, population size
    @param num_cpu_cores: int, number of cores to use for parallelization of the objective function
    @param parallel: bool, whether to run the objective function in parallel
    @param total_budget: float, the total budget that is to be distributed along the nodes of the graph
    @param statistic: function object,
    @param eval_function: function object,
    @param bounds: list,
    @param n_simulations: int,
    @param node_indices: list,
    @param n_nodes: int,
    @param initial_population: numpy array,
    @param sigma: float,
    @param max_iterations: int,
    @return: the parameters for the best solution found by the optimizer,
            the best solutions found by the optimizer during the run, the time take for the optimization
    """

    f_kwargs = {'n_simulations': n_simulations, 'node_indices': node_indices, 'n_nodes': n_nodes,
                'eval_function': eval_function, 'statistic': statistic, 'total_budget': total_budget,
                'parallel': parallel, 'num_cpu_cores': num_cpu_cores}

    es = cma.CMAEvolutionStrategy(initial_population, sigma0=sigma,
                                  inopts={'maxiter': max_iterations,
                                          'verbose': -8,
                                          'bounds': bounds,
                                          'popsize': popsize})

    t_start = time.time()
    time_for_optimization = []
    best_solution_history = []
    best_solution_stderr_history = []
    n = 0
    while not es.stop():
        # sample a new population of solutions
        solutions = es.ask()
        # evaluate all solutions on the objective function, return only the mean (omit stderr)
        f_solution = [bo_cma_objective_function(s, **f_kwargs)[0] for s in tqdm(solutions, leave=False)]
            # parallelization is non-trivial, as the objective function is already parallelized and nested
            # parallelization is not allowed by python
        # use the solution and evaluation to update cma-es parameters (covariance-matrix ...)
        es.tell(solutions, f_solution)
        # evaluate the current best parameter and save mean + standard error
        best_s, best_s_stderr = bo_cma_objective_function(es.result.xbest, **f_kwargs)
        best_solution_history.append(best_s)
        best_solution_stderr_history.append(best_s_stderr)

        es.logger.add()
        es.disp()   # prints the progress of the optimizer
        time_for_optimization.append((time.time() - t_start) / 60)

        if save_test_strategies:
            np.save(os.path.join(save_test_strategies_path, f'test_strategy_{n}'), solutions)
            n += 1

    best_parameter = es.result.xbest

    return best_parameter, best_solution_history, best_solution_stderr_history, time_for_optimization


def bo_cma_objective_function(x, n_simulations, node_indices, n_nodes, eval_function,
                           statistic, total_budget, parallel, num_cpu_cores):
    """
    An optimizeable objective function.
    Maps a lower dimensional x to their corresponding indices in the input vector of the given objective function.


    @param num_cpu_cores: int, number of cpu cores to use for parallelization
    @param parallel: bool, whether to run the SI-simulation in parallel
    @param total_budget: int, total budget to be allocated to the sentinels
    @param statistic: function object, the statistic to be used for the SI-simulation
    @param x: list of floats, the parameters to be optimized
    @param eval_function: function object,
    @param n_simulations: int, number of times the objective function will run a simulation for averaging the output
    @param node_indices: list, indices of x in the higher dimensional x
    @param n_nodes: int, dimension of the input of the objective function
    @return: two floats, the SI-simulation result and standard error
    """
    # rescale strategy such that it satisfies sum constraint
    x = total_budget * bo_softmax(x)

    x = bo_map_low_dim_x_to_high_dim(x, n_nodes, node_indices)

    y, stderr = eval_function(x,
                              n_simulations=n_simulations,
                              parallel=parallel,
                              num_cpu_cores=num_cpu_cores,
                              statistic=statistic)
    return y, stderr


if __name__ == '__main__':
    # Prints out the all hyperparameters for CMA
    for k in cma.CMAOptions():
        print(k, cma.CMAOptions()[k])
