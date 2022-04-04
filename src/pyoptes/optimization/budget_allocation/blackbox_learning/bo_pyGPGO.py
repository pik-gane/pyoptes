import time
import numpy as np

from .utils import map_low_dim_x_to_high_dim

from pyGPGO.acquisition import Acquisition
from pyGPGO.covfunc import squaredExponential
from pyGPGO.surrogates.GaussianProcess import GaussianProcess

from .custom_GPGO import GPGO


def bo_pyGPGO(prior, max_iterations, n_simulations, node_indices, n_nodes, eval_function,
              total_budget, parallel, cpu_count, acquisition_function, use_prior=True):
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
    @param cpu_count:
    @return:
    """
    # variables in uppercase are used in the objective function
    # this is just done to keep them visually distinct from variables used inside the function
    # as the optimizer doesn't allow passing of arguments to a function

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
                n_jobs=cpu_count,
                f_kwargs={'node_indices': node_indices, 't_start': t_start, 'total_budget': total_budget,
                          'n_nodes': n_nodes, 'eval_function': eval_function, 'n_simulations': n_simulations,
                          'parallel': parallel, 'cpu_count': cpu_count, 'time_for_optimization': time_for_optimization})
    gpgo.run(max_iter=max_iterations,
             prior=prior,
             use_prior=use_prior)

    # compute boundaries for each y by adding/subtracting the standard-error
    yu = [-y - gpgo.stderr[y] for y in gpgo.history]
    yo = [-y + gpgo.stderr[y] for y in gpgo.history]

    # gpg.history contains the best y of the gp at each iteration
    # reverse the sign change of optimizer history to get a more readable plot
    return [gpgo.getResult(), gpgo.stderr[gpgo.getResult()[1]]], -np.array(gpgo.history), \
           time_for_optimization, np.array(gpgo.time_history), np.array([yu, yo])


def pyGPGO_objective_function(x, node_indices, t_start, total_budget, n_nodes, eval_function,
                              n_simulations, parallel, cpu_count, time_for_optimization):
    """

    @param time_for_optimization:
    @param x:
    @param t_start:
    @param n_simulations:
    @param node_indices:
    @param n_nodes:
    @param eval_function:
    @param total_budget:
    @param parallel:
    @param cpu_count:
    @return:
    """

    assert np.shape(x) == np.shape(node_indices)

    time_for_optimization.append((time.time()-t_start)/60)

    # rescale strategy such that it satisfies sum constraint
    x = total_budget * np.exp(x) / sum(np.exp(x))

    x = map_low_dim_x_to_high_dim(x, n_nodes, node_indices)

    # GPGO maximises a function, therefore the minus is added in front of the eval_function so the function is minimised
    y, stderr = eval_function(x,
                              n_simulations=n_simulations,
                              parallel=parallel,
                              num_cpu_cores=cpu_count)
    return -y, stderr


if __name__ == '__main__':
    print('success')
