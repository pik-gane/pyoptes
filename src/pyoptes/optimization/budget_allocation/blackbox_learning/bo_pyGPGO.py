import time
import numpy as np

from .utils import map_low_dim_x_to_high_dim

from pyGPGO.acquisition import Acquisition
from pyGPGO.covfunc import squaredExponential
from pyGPGO.surrogates.GaussianProcess import GaussianProcess

from .custom_GPGO import GPGO


def bo_pyGPGO(prior, max_iterations, n_simulations, node_indices, n_nodes, eval_function,
              total_budget, parallel, num_cpu_cores, acquisition_function, statistic, use_prior=True):
    """
    Run GPGO, a Bayesian optimization algorithm with a gaussian process surrogate.

    @param use_prior: bool, sets whether the surrogate function is pre-fit with a prior or random samples
    @param prior: list of arrays, the prior for the surrogate function
    @param acquisition_function: string, defines the acquisition function to be used.
    @param max_iterations: int, maximum number of iterations for GPGO to run
    @param n_simulations: int, number of simulations the SI-simulation is run with
    @param node_indices: list of ints, the indices of the sentinels (nodes that are to be optimized)
    @param n_nodes: int, number of nodes in the network
    @param eval_function: function, the function to be optimized (SI-simulation)
    @param total_budget: int, total budget to be allocated to the sentinels
    @param parallel: bool, whether to run the SI-simulation in parallel
    @param num_cpu_cores: int, number of cpu cores to use for the SI-simulation
    @return:
        1. best_test_strategy: best strategy found over all iterations
        2. best_solution_history: list of best strategies found over all iterations
        3. stderr_history: list of standard errors of the best strategies found over all iterations
        4. time_history: time take to run each iteration (cumulative

    """

    sexp = squaredExponential()
    gp = GaussianProcess(sexp)
    acq = Acquisition(mode=acquisition_function)

    # create a list of the parameters to be optimized with their bounds
    parameters = {}
    for i in range(len(node_indices)):
        parameters[f"x{i}"] = ('cont', [0.0, float(total_budget)])

    gpgo = GPGO(surrogate=gp,
                acquisition=acq,
                f=pyGPGO_objective_function,
                parameter_dict=parameters,
                n_jobs=num_cpu_cores,
                f_kwargs={'node_indices': node_indices, 'total_budget': total_budget,
                          'n_nodes': n_nodes, 'eval_function': eval_function,
                          'n_simulations': n_simulations, 'parallel': parallel,
                          'num_cpu_cores': num_cpu_cores, 'statistic': statistic})

    gpgo.run(max_iter=max_iterations,
             prior=prior,
             use_prior=use_prior)

    best_test_strategy = gpgo.getResult()[0]
    # gpgo.history contains the best y of the gp at each iteration
    # reverse the sign change of optimizer history to get a more readable plot
    best_solution_history = -np.array(gpgo.history)
    # get the correct stderr of the solutions at each timestep
    stderr_history = [gpgo.stderr[y] for y in gpgo.history]

    return best_test_strategy, best_solution_history, stderr_history, gpgo.time_for_optimization


def pyGPGO_objective_function(x, node_indices, total_budget, n_nodes, eval_function,
                              n_simulations, parallel, num_cpu_cores, statistic):
    """
    Objective function for GPGO that is to be optimized.
    Is a wrapper around the SI-simulation.
    Automatically scales the sum of the budget of x such, that it is lower or equal to the total budget.
    The sign of the SI-simulation output is changed to get GPGO to minimize the function value
    @param statistic: function, the statistic to be used for the SI-simulation
    @param x: list of floats, the parameters to be optimized
    @param n_simulations: int, number of simulations the SI-simulation is run with
    @param node_indices: list of ints, the indices of the sentinels (nodes that are to be optimized)
    @param n_nodes: int, number of nodes in the network
    @param eval_function: SI-simulation
    @param total_budget: int, total budget to be allocated to the sentinels
    @param parallel: bool, whether to run the SI-simulation in parallel
    @param num_cpu_cores: int, number of cpu cores to use for the SI-simulation
    @return: two floats, the SI-simulation result and standard error
    """
    # TODO fix GPGO breaking when using the prior + sentinels less the n_nodes

    # rescale strategy such that it satisfies sum constraint
    x = total_budget * np.exp(x) / sum(np.exp(x))

    x = map_low_dim_x_to_high_dim(x, n_nodes, node_indices)

    y, stderr = eval_function(x,
                              n_simulations=n_simulations,
                              parallel=parallel,
                              num_cpu_cores=num_cpu_cores,
                              statistic=statistic)
    # GPGO maximises a function, therefore the minus is added in front of the eval_function so the function is minimised
    return -y, stderr


if __name__ == '__main__':
    print('success')
