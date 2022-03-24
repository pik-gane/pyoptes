'''
Rewrite of the GPGO class from pyGPGO.
Adjusted to support

    Adaption of the GPGO-class. Extended to support fitting the surrogate function with a prior.
    run has been changed to reduce the verbosity of the class.

'''

import time
from tqdm import tqdm
from multiprocessing import cpu_count

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import minimize

# TODO write check for availability of n_jobs
class GPGO:
    def __init__(self, surrogate, acquisition, f, parameter_dict, n_jobs=15, f_kwargs={}):
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

        self.time_history = []
        self.stderr = {}

        self.f_kwargs = f_kwargs

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
        Performs initial evaluations before fitting GP.

        Parameters
        ----------
        n_eval: int
            Number of initial evaluations to perform. Default is 3.

        """
        self.X = np.empty((n_eval, len(self.parameter_key)))
        self.y = np.empty((n_eval,))
        for i in range(n_eval):
            s_param = self._sampleParam()
            self.X[i] = s_param
            self.y[i], stderr = self.f(s_param, **self.f_kwargs)
        self.GP.fit(self.X, self.y)

        self.tau = np.max(self.y)
        self.tau = np.round(self.tau, decimals=8)

        self.history.append(self.tau)
        self.stderr[self.tau] = stderr

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

        self.best = x_best[np.argmin(f_best)]

    def updateGP(self):
        """
        Updates the internal model with the next acquired point and its evaluation.
        """
        # TODO how kw is created looks inefficient
        kw = {param: self.best[i]
              for i, param in enumerate(self.parameter_key)}
        param = np.array(list(kw.values()))

        f_new, stderr = self.f(param, **self.f_kwargs)    # returns the y corresponding to a test strategy
        self.GP.update(np.atleast_2d(self.best), np.atleast_1d(f_new))

        self.tau = np.max(self.GP.y)    # self.GP "saves" the y from the objective f,
        self.tau = np.round(self.tau, decimals=8)

        # test strategies return the same y for f and self.GP (+/- the standarderror)
        # GP.y is just a list
        self.history.append(self.tau)
        self.stderr[self.tau] = stderr

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

    def _fitGP(self, prior):
        """

        @param prior: list of test strategies
        """
        Y = []
        X = []
        for x in prior:
            X.append(x)
            y, stderr = self.f(x, **self.f_kwargs)
            Y.append(y)

        self.GP.fit(np.array(X), np.array(Y))
        self.tau = np.max(y)
        self.tau = np.round(self.tau, decimals=8)

        self.history.append(self.tau)
        self.stderr[self.tau] = stderr

    def run(self, max_iter=10, init_evals=3, prior=None, use_prior=False):
        """
        Runs the Bayesian Optimization procedure.

        Parameters
        ----------
        max_iter: int
            Number of iterations to run. Default is 10.
        init_evals: int
            Initial function evaluations before fitting a GP. Default is 3.
        """
        if not use_prior:
            print('Running GPGO with surrogate function fitted on randomly sampled points\n')
            self.init_evals = init_evals
            self._firstRun(self.init_evals)
        else:
            print('Running GPGO with surrogate function fitted on prior.\n'
                  'Fitting the GP takes about a minute (depending on the size of the prior)\n')
            self._fitGP(prior)

        for _ in tqdm(range(max_iter)):
            time_ac = time.time()
            self._optimizeAcq()
            time_ac = time.time() - time_ac
            time_gp = time.time()
            self.updateGP()
            time_gp = time.time() - time_gp
            self.time_history.append([time_ac/60, time_gp/60])