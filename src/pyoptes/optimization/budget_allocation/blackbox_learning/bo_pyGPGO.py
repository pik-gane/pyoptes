import time
import numpy as np
from collections import OrderedDict

from .utils import map_low_dim_x_to_high_dim

from pyGPGO.acquisition import Acquisition
from pyGPGO.covfunc import squaredExponential
from pyGPGO.surrogates.GaussianProcess import GaussianProcess

from .custom_GPGO import GPGO


def bo_pyGPGO(prior, max_iterations, n_simulations, node_indices, n_nodes, eval_function,
              statistic, total_budget, parallel, cpu_count, acquisition_function,
              use_prior=True):
    """

    @param use_prior:
    @param prior:
    @param acquisition_function:
    @param max_iterations:
    @param n_simulations:
    @param node_indices:
    @param n_nodes:
    @param eval_function:
    @param statistic:
    @param total_budget:
    @param parallel:
    @param cpu_count:
    @return:
    """
    # variables in uppercase are used in the objective function
    # this is just done to keep them visually distinct from variables used inside the function
    # as the optimizer doesn't allow passing of arguments to a function
    EVAL_FUNCTION = eval_function
    N_SIMULATIONS = n_simulations
    NODE_INDICES = node_indices
    STATISTIC = statistic
    TOTAL_BUDGET = total_budget
    N_NODES = n_nodes
    PARALLEL = parallel
    CPU_COUNT = cpu_count

    T_START = time.time()
    time_for_optimization = []

    def pyGPGO_objective_function(**kwargs):
        """

        @param kwargs:
        @return:
        """
        x = np.array(list(kwargs.values()))
        assert np.shape(x) == np.shape(NODE_INDICES)

        time_for_optimization.append((time.time()-T_START)/60)

        # rescale strategy such that it satisfies sum constraint
        x = TOTAL_BUDGET * np.exp(x) / sum(np.exp(x))

        x = map_low_dim_x_to_high_dim(x, N_NODES, NODE_INDICES)

        # GPGO maximises a function, therefore the minus is added in front of the eval_function
        y, stderr = EVAL_FUNCTION(x,
                                  n_simulations=N_SIMULATIONS,
                                  # statistic=STATISTIC,
                                  parallel=PARALLEL,
                                  num_cpu_cores=CPU_COUNT)
        return -y, stderr

    sexp = squaredExponential()
    gp = GaussianProcess(sexp)
    acq = Acquisition(mode=acquisition_function)

    # create a list of the parameters to be optimized with their bounds
    parameters = {}
    for i in range(len(node_indices)):
        parameters[f"x{i}"] = ('cont', [0.0, float(total_budget)])

    aux = None
    if use_prior:
        aux = [OrderedDict((f'x{i}', v) for i, v in enumerate(x)) for x in prior]
    gpgo = GPGO(surrogate=gp,
                acquisition=acq,
                f=pyGPGO_objective_function,
                parameter_dict=parameters)
    gpgo.run(max_iter=max_iterations,
             prior=aux,
             use_prior=use_prior)

    # compute boundaries for each y by adding/subtracting the standard-error
    yu = [-y - gpgo.stderr[y] for y in gpgo.history]
    yo = [-y + gpgo.stderr[y] for y in gpgo.history]

    # gpg.history contains the best y of the gp at each iteration
    # reverse the sign change of optimizer history to get a more readable plot
    return [gpgo.getResult(), gpgo.stderr[gpgo.getResult()[1]]], -np.array(gpgo.history), \
           time_for_optimization, np.array(gpgo.time_history), np.array([yu, yo])


if __name__ == '__main__':
    print('success')
