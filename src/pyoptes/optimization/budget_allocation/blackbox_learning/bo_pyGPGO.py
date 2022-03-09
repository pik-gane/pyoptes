
import os
import time
import numpy as np
import pylab as plt

from .utils import map_low_dim_x_to_high_dim

from pyGPGO.GPGO import GPGO
from pyGPGO.acquisition import Acquisition
from pyGPGO.covfunc import squaredExponential
from pyGPGO.surrogates.GaussianProcess import GaussianProcess


def bo_pyGPGO(max_iterations, n_simulations, node_indices, n_nodes, eval_function,
              path_experiment, statistic, total_budget, parallel, cpu_count, log_level):
    """

    @param max_iterations:
    @param n_simulations:
    @param node_indices:
    @param n_nodes:
    @param eval_function:
    @param path_experiment:
    @param statistic:
    @param total_budget:
    @param parallel:
    @param cpu_count:
    @return:
    """
    # variables in upppercase are used in the objective function
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
    MAX_ITERATIONS = max_iterations

    LOG_ITERATOR = [0]  # has to be a list, normal ints are not updated in this
    # log_level defines the percentage of iterations for which a log message appears
    # LOG_INTERVAL is then the number of iteration between two log messages
    LOG_INTERVAL = int(max_iterations/100*log_level)
    print(LOG_INTERVAL)

    T_START = time.time()

    def pyGPGO_objective_function(**kwargs):

        x = np.array(list(kwargs.values()))
        assert np.shape(x) == np.shape(NODE_INDICES)

        if LOG_ITERATOR[0] != 0 and LOG_ITERATOR[0] % LOG_INTERVAL == 0:
            print(f'\nIteration: {LOG_ITERATOR[0]}/{MAX_ITERATIONS}. '
                  f'Minutes elapsed since start: {(time.time()-T_START)/60}')
        LOG_ITERATOR[0] = LOG_ITERATOR[0]+1

        # create a dummy vector to be filled with the values of x at the appropriate indices
        x = TOTAL_BUDGET * np.exp(x) / sum(np.exp(x))

        x = map_low_dim_x_to_high_dim(x, N_NODES, NODE_INDICES)

        return -EVAL_FUNCTION(x, n_simulations=N_SIMULATIONS, statistic=STATISTIC,
                              parallel=PARALLEL, num_cpu_cores=CPU_COUNT)

    sexp = squaredExponential()
    gp = GaussianProcess(sexp)
    acq = Acquisition(mode='ExpectedImprovement')

    parameters = {}
    for i in range(len(node_indices)):
        parameters[f"x{i}"] = ('cont', [0.0, float(total_budget)])

    np.random.seed(23)
    gpgo = GPGO(gp, acq, pyGPGO_objective_function, parameters)
    gpgo.run(max_iter=max_iterations)

    plt.plot(range(len(gpgo.history)), gpgo.history)
    plt.title('SMAC')
    plt.xlabel('Iteration')
    plt.ylabel('SI-model output')
    plt.savefig(os.path.join(path_experiment, 'GPGO_plot.png'))

    return gpgo.getResult()


if __name__ == '__main__':
    bo_pyGPGO()
    print('success')
