import os
import time
import numpy as np
import pylab as plt
from collections import OrderedDict

from .utils import map_low_dim_x_to_high_dim

from pyGPGO.GPGO import GPGO
from pyGPGO.acquisition import Acquisition
from pyGPGO.covfunc import squaredExponential
from pyGPGO.surrogates.GaussianProcess import GaussianProcess


class ex_GPGO(GPGO):
    def fit_gp(self, initial_X):
        y=[]
        X=[]
        for x in initial_X:
            X.append(list(x.values()))
            y.append(self.f(**x))

        self.GP.fit(np.array(X), np.array(y))
        self.tau = np.max(y)
        self.history.append(self.tau)

    def run(self, max_iter=10, init_evals=3, resume=False):
        """
        Runs the Bayesian Optimization procedure.

        Parameters
        ----------
        max_iter: int
            Number of iterations to run. Default is 10.
        """

        for iteration in range(max_iter):
            self._optimizeAcq()
            self.updateGP()


def bo_pyGPGO(initial_X, max_iterations, n_simulations, node_indices, n_nodes, eval_function,
              path_experiment, statistic, total_budget, parallel, cpu_count, log_level):
    """

    @param log_level:
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
    MAX_ITERATIONS = max_iterations

    LOG_ITERATOR = [1]  # has to be a list, normal ints are not updated in this
    # log_level defines the percentage of iterations for which a log message appears
    # LOG_INTERVAL is then the number of iteration between two log messages
    LOG_INTERVAL = int(max_iterations*(log_level*10)/100)

    T_START = time.time()
    time_for_optimization = []

    def pyGPGO_objective_function(**kwargs):
        """

        @param kwargs:
        @return:
        """
        x = np.array(list(kwargs.values()))
        assert np.shape(x) == np.shape(NODE_INDICES)

        t = (time.time()-T_START)/60
        time_for_optimization.append(t)

        # TODO find a way to deal with initial GP fitting messing up log
        if LOG_ITERATOR[0]-len(initial_X) < 1 and LOG_ITERATOR[0] % LOG_INTERVAL == 0:
            print(f'\n'
                  f'-------------------------------------------\n'
                  f'Iteration: {LOG_ITERATOR[0]}/{MAX_ITERATIONS}. '
                  f'Minutes elapsed since start: {t}\n'
                  f'-------------------------------------------\n')
        LOG_ITERATOR[0] = LOG_ITERATOR[0]+1

        # rescale strategy such that it satifies sum constraint
        x = TOTAL_BUDGET * np.exp(x) / sum(np.exp(x))

        x = map_low_dim_x_to_high_dim(x, N_NODES, NODE_INDICES)

        # GPGO maximises a function, therefore the minus is added in front of the eval_function
        return -EVAL_FUNCTION(x,
                              n_simulations=N_SIMULATIONS,
                              statistic=STATISTIC,
                              parallel=PARALLEL,
                              num_cpu_cores=CPU_COUNT)

    sexp = squaredExponential()
    gp = GaussianProcess(sexp)
    acq = Acquisition(mode='ExpectedImprovement')

    parameters = {}
    for i in range(len(node_indices)):
        parameters[f"x{i}"] = ('cont', [0.0, float(total_budget)])

    gpgo = GPGO(surrogate=gp,
                acquisition=acq,
                f=pyGPGO_objective_function,
                parameter_dict=parameters)
    gpgo.run(max_iter=max_iterations)

    # aux = [OrderedDict((f'x{i}', v) for i, v in enumerate(l)) for l in initial_X]
    #
    # gpgo = ex_GPGO(surrogate=gp,
    #                acquisition=acq,
    #                f=pyGPGO_objective_function,
    #                parameter_dict=parameters)
    # gpgo.fit_gp(aux)
    # gpgo.run(max_iter=max_iterations)

    # reverse the sign change of optimizer history to get a more readable plot
    plt.plot(range(len(gpgo.history)), -np.array(gpgo.history))
    plt.title(f'GPGO, {n_nodes} nodes, {len(node_indices)} sentinels')
    plt.xlabel('Iteration')
    plt.ylabel('SI-model output')
    plt.savefig(os.path.join(path_experiment, 'GPGO_plot.png'))

    plt.clf()
    plt.plot(range(len(time_for_optimization)), time_for_optimization)
    plt.title(f'Time for objective function evaluation, {n_nodes} nodes, {len(node_indices)} sentinels')
    plt.xlabel('Iteration')
    plt.ylabel('Time in minutes')
    plt.savefig(os.path.join(path_experiment, 'time_for_optimization.png'))
    return gpgo.getResult()


if __name__ == '__main__':
    print('success')
