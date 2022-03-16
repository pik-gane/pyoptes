import time
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

from .utils import map_low_dim_x_to_high_dim

from pyGPGO.GPGO import GPGO
from pyGPGO.acquisition import Acquisition
from pyGPGO.covfunc import squaredExponential
from pyGPGO.surrogates.GaussianProcess import GaussianProcess


# TODO maybe rewrite GPGO to be able to work when the target_function returns the standard deviation
# TODO replace log by tqdm
class GPGO(GPGO):
    '''
    Adaption of the GPGO-class. Extended to support fitting the surrogate function with a prior.
    run has been changed to reduce the verbosity of the class.
    '''
    def fit_gp(self, prior):
        """

        @param initial_X:
        """
        y = []
        X = []
        for x in prior:
            X.append(list(x.values()))
            y.append(self.f(**x))

        self.GP.fit(np.array(X), np.array(y))
        self.tau = np.max(y)
        self.history.append(self.tau)

    def run_with_prior(self, max_iter=10):
        """
        Runs the Bayesian Optimization procedure.
        fit_gp has to be run before this.
        Parameters
        ----------
        max_iter: int
            Number of iterations to run. Default is 10.
        """

        for _ in tqdm(range(max_iter)):
            self._optimizeAcq()
            self.updateGP()

    def run(self, max_iter=10, init_evals=3, resume=False):
        """
        Runs the Bayesian Optimization procedure.

        Parameters
        ----------
        max_iter: int
            Number of iterations to run. Default is 10.
        init_evals: int
            Initial function evaluations before fitting a GP. Default is 3.
        resume: bool
            Whether to resume the optimization procedure from the last evaluation. Default is `False`.
        """
        if not resume:
            self.init_evals = init_evals
            self._firstRun(self.init_evals)
        for _ in tqdm(range(max_iter)):
            self._optimizeAcq()
            self.updateGP()


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
    acq = Acquisition(mode=acquisition_function)

    # create a list of the parameters to be optimized with their bounds
    parameters = {}
    for i in range(len(node_indices)):
        parameters[f"x{i}"] = ('cont', [0.0, float(total_budget)])

    if use_prior:
        print('Running GPGO with surrogate function fitted on prior.\n'
              'Fitting the GP takes about a minute (depending on the size of the prior)\n')
        aux = [OrderedDict((f'x{i}', v) for i, v in enumerate(x)) for x in prior]

        gpgo = GPGO(surrogate=gp,
                    acquisition=acq,
                    f=pyGPGO_objective_function,
                    parameter_dict=parameters)
        gpgo.fit_gp(aux)
        gpgo.run_with_prior(max_iter=max_iterations)
    else:
        print('Running GPGO with surrogate function fitted on randomly sampled points\n')
        gpgo = GPGO(surrogate=gp,
                    acquisition=acq,
                    f=pyGPGO_objective_function,
                    parameter_dict=parameters)
        gpgo.run(max_iter=max_iterations)

    # reverse the sign change of optimizer history to get a more readable plot
    return gpgo.getResult(), -np.array(gpgo.history), time_for_optimization


if __name__ == '__main__':
    print('success')
