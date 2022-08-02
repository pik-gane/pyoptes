# TODO rewrite this to make the changes more clear
'''
Adaption of the GPGO-class. Extended to support fitting the surrogate function with a prior.
The class also returns the stderr of objective function calls, as well as the time spent for the optimization.
'''

import os
import time
from tqdm import tqdm
from multiprocessing import cpu_count

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import minimize


class GPGO:
    def __init__(self, surrogate, acquisition, f, parameter_dict, n_jobs=15, f_kwargs={},
                 save_test_strategies=False, save_test_strategies_path=None):
        """
        Bayesian Optimization class.

        Parameters
        ----------
        surrogate: Surrogate model instance
            Gaussian Process surrogate model instance.
        acquisition: Acquisition instance
            Acquisition instance.
        f: fun
            Function to maximize over parameters specified by `parameter_dict`.
        parameter_dict: dict
            Dictionary specifying parameter, their type and bounds.
        n_jobs: int. Default 1
            Parallel threads to use during acquisition optimization.

        Attributes
        ----------
        parameter_key: list
            Parameters to consider in optimization
        parameter_type: list
            Parameter types.
        parameter_range: list
            Parameter bounds during optimization
        history: list
            Target values evaluated along the procedure.
        """
        self.GP = surrogate
        self.A = acquisition
        self.f = f
        self.parameters = parameter_dict

        # check whether the specified number of cpus are available
        if n_jobs > cpu_count():
            self.n_jobs = cpu_count()
        else:
            self.n_jobs = n_jobs

        self.parameter_key = list(parameter_dict.keys())
        self.parameter_value = list(parameter_dict.values())
        self.parameter_type = [p[0] for p in self.parameter_value]
        self.parameter_range = [p[1] for p in self.parameter_value]

        self.history = []

        self.time_start = time.time()
        self.time_for_optimization = []
        self.time_acqui_predict = []



        self.stderr = {}

        self.f_kwargs = f_kwargs

        self.n = 0
        self.save_test_strategies = save_test_strategies
        self.save_test_strategies_path = save_test_strategies_path

    def _sampleParam(self):
        """
        Randomly samples parameters over bounds.

        Returns
        -------
        dict:
            A random sample of specified parameters.
        """
        d = []
        for index, param in enumerate(self.parameter_key):
            if self.parameter_type[index] == 'int':
                d.append(np.random.randint(
                    self.parameter_range[index][0], self.parameter_range[index][1]))
            elif self.parameter_type[index] == 'cont':
                d.append(np.random.uniform(
                    self.parameter_range[index][0], self.parameter_range[index][1]))
            else:
                raise ValueError('Unsupported variable type.')
        return d

    def _firstRun(self, n_eval=3):
        """
        Performs initial evaluations on random samples of the parameters before fitting GP.

        Parameters
        ----------
        n_eval: int
            Number of initial evaluations to perform. Default is 3.

        """
        X = np.empty((n_eval, len(self.parameter_key)))
        Y = np.empty((n_eval,))
        Y_stderr = np.empty((n_eval,))
        for i in range(n_eval):
            s_param = self._sampleParam()
            X[i] = s_param
            Y[i], Y_stderr[i] = self.f(s_param, **self.f_kwargs)

        self.GP.fit(X, Y)

        # get the best y and corresponding stderr
        i = np.argmax(Y)
        self.tau = Y[i]
        tau_stderr = Y_stderr[i]
        self.tau = np.round(self.tau, decimals=8)

        self.history.append(self.tau)
        self.stderr[self.tau] = tau_stderr

    def _acqWrapper(self, xnew):
        """
        Evaluates the acquisition function on a point.

        Parameters
        ----------
        xnew: np.ndarray, shape=((len(self.parameter_key),))
            Point to evaluate the acquisition function on.

        Returns
        -------
        float
            Acquisition function value for `xnew`.

        """
        new_mean, new_var = self.GP.predict(xnew, return_std=True)
        new_std = np.sqrt(new_var + 1e-6)

        return -self.A.eval(self.tau, new_mean, new_std)

    def _optimizeAcq(self, method='L-BFGS-B', n_start=100):
        """
        Optimizes the acquisition function using a multistart approach.

        Parameters
        ----------
        method: str. Default 'L-BFGS-B'.
            Any `scipy.optimize` method that admits bounds and gradients. Default is 'L-BFGS-B'.
        n_start: int.
            Number of starting points for the optimization procedure. Default is 100.

        """
        # TODO check which part here is the slowest
        # TODO maybe test different acqui-functions
        start_points_arr = np.array([self._sampleParam() for i in range(n_start)])
        if self.save_test_strategies:
            np.save(os.path.join(self.save_test_strategies_path, f'test_strategy_{self.n}'), start_points_arr)
            self.n += 1

        x_best = np.empty((n_start, len(self.parameter_key)))
        f_best = np.empty((n_start,))
        if self.n_jobs == 1:
            for index, start_point in enumerate(start_points_arr):
                res = minimize(self._acqWrapper, x0=start_point, method=method,
                               bounds=self.parameter_range)
                x_best[index], f_best[index] = res.x, np.atleast_1d(res.fun)[0]
        else:
            opt = Parallel(n_jobs=self.n_jobs)(delayed(minimize)(self._acqWrapper,
                                                                 x0=start_point,
                                                                 method=method,
                                                                 bounds=self.parameter_range) for start_point in
                                               start_points_arr)
            x_best = np.array([res.x for res in opt])
            f_best = np.array([np.atleast_1d(res.fun)[0] for res in opt])

        self.current_best_measurement = x_best[np.argmin(f_best)]

    def updateGP(self):
        """
        Updates the internal model with the next acquired point and its evaluation.
        """
        kw = {param: self.current_best_measurement[i]
              for i, param in enumerate(self.parameter_key)}
        param = np.array(list(kw.values()))

        # f_new is always the newest measurement for the objective function, not necessarily the best one
        f_new, stderr_f_new = self.f(param, **self.f_kwargs)    # returns the y corresponding to a test strategy
        self.GP.update(np.atleast_2d(self.current_best_measurement), np.atleast_1d(f_new))

        # add new stderr to the stderr dictionary. This ensures that there is always a stderr for each measurement
        f_new = np.round(f_new, decimals=8)
        self.stderr[f_new] = stderr_f_new

        # get the current optimum of the GP
        self.tau = np.max(self.GP.y) # self.GP "saves" the y from the objective f,
        self.tau = np.round(self.tau, decimals=8)
        self.history.append(self.tau)

    def getResult(self):
        """
        Prints best result in the Bayesian Optimization procedure.

        Returns
        -------
        float
            Best function evaluation.

        """
        argtau = np.argmax(self.GP.y)
        opt_x = self.GP.X[argtau]
        res_d = []
        for i, (key, param_type) in enumerate(zip(self.parameter_key, self.parameter_type)):
            if param_type == 'int':
                res_d.append(int(opt_x[i]))
            else:
                res_d.append(opt_x[i])
        return res_d, self.tau

    def _fitGP(self, prior, prior_y, prior_stderr):
        """

        @param prior: list of test strategies
        @param prior_y: list of corresponding function evaluations
        @param prior_stderr: list of corresponding stderr
        """

        prior_y = -1 * np.array(prior_y)
        self.GP.fit(np.array(prior), np.array(prior_y))

        # get the best y and corresponding stderr
        i = np.argmax(prior_y)
        self.tau = np.round(prior_y[i], decimals=8)
        tau_stderr = prior_stderr[i]

        self.history.append(self.tau)
        self.stderr[self.tau] = tau_stderr

    def run(self, max_iter=10, init_evals=3, use_prior=False,
            prior=None, prior_y=None, prior_stderr=None):
        """
        Runs the Bayesian Optimization procedure.
        @param prior_y:
        @param prior_stderr:
        @param max_iter: maximum number of iterations for GPGO
        @param init_evals:  number of random samples for fitting the GP
        @param prior: list of test strategies
        @param use_prior: boolean to use the prior
        """

        if not use_prior:
            print('Running GPGO with surrogate function fitted on randomly sampled points\n')
            self._firstRun(init_evals)
        else:
            print('Running GPGO with surrogate function fitted on prior.\n'
                  'Fitting the GP takes about a minute (depending on the size of the prior and the size of the graph)')
            self._fitGP(prior=prior,
                        prior_y=prior_y,
                        prior_stderr=prior_stderr)
            print('GP fitted.')

        # print(f'Running GPGO for {max_iter} iterations.')
        for _ in tqdm(range(max_iter), leave=False):

            self._optimizeAcq()
            self.updateGP()
            time_optim = time.time() - self.time_start
            self.time_for_optimization.append(time_optim/60)
