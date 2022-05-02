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

    @param use_prior:
    @param prior:
    @param acquisition_function:
    @param max_iterations:
    @param n_simulations:
    @param node_indices:
    @param n_nodes:
    @param eval_function:
    @param total_budget:
    @param parallel:
    @param num_cpu_cores:
    @return:
    """

    t_start = time.time()
    time_for_optimization = []

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

    @param x:
    @param n_simulations:
    @param node_indices:
    @param n_nodes:
    @param eval_function:
    @param total_budget:
    @param parallel:
    @param num_cpu_cores:
    @return:
    """
    # TODO fix GPGO breaking when using the prior + sentinels less the n_nodes
    assert np.shape(x) == np.shape(node_indices)

    # rescale strategy such that it satisfies sum constraint
    x = total_budget * np.exp(x) / sum(np.exp(x))

    x = map_low_dim_x_to_high_dim(x, n_nodes, node_indices)

    # GPGO maximises a function, therefore the minus is added in front of the eval_function so the function is minimised
    y, stderr = eval_function(x,
                              n_simulations=n_simulations,
                              parallel=parallel,
                              num_cpu_cores=num_cpu_cores,
                              statistic=statistic)
    return -y, stderr


if __name__ == '__main__':
    print('success')
