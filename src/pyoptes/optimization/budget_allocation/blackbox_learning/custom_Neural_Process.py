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

from .neural_process.neural_process import NeuralProcess
from .neural_process.training import NeuralProcessTrainer
from .neural_process.utils_np import context_target_split, TrainingDataset
from torch.utils.data import DataLoader
import torch


class NP:
    def __init__(self, acquisition, f, parameter_dict, prior_x, prior_y, prior_stderr,
                 n_jobs=15, f_kwargs={},
                 save_test_strategies=False, save_test_strategies_path=None,):
        """
        Bayesian Optimization class.

        Parameters
        ----------
        acquisition: Acquisition instance
            Acquisition instance.
        f: fun
            Function to maximize over parameters specified by `parameter_dict`.
        parameter_dict: dict
            Dictionary specifying parameter, their type and bounds.
        n_jobs: int. Default 1
            Parallel threads to use during acquisition optimization.

        """
        # TODO should come from outside
        x_dim = len(f_kwargs['node_indices'])  # dimension of the objective function, equal to the number of sentinels
        print('x_dim: ', x_dim)
        y_dim = 1
        r_dim = 50  # Dimension of representation of context points
        z_dim = 50  # Dimension of sampled latent variable
        h_dim = 50  # Dimension of hidden layers in encoder and decoder

        self.num_context = 3  # num_context + num_target has to be lower than num_samples
        self.num_target = 3

        self.NP = NeuralProcess(x_dim, y_dim, r_dim, z_dim, h_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = torch.optim.Adam(self.NP.parameters(), lr=3e-4)

        self.prior_x = prior_x
        self.prior_y = prior_y
        self.prior_stderr = prior_stderr

        self.x = torch.tensor(prior_x).unsqueeze(0).float()
        self.y = torch.tensor(prior_y).unsqueeze(1).unsqueeze(0).float()

        # -----------------------------------------------------------------------------------------------------------

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
        new_mean, new_std = self.NP_predict(xnew)

        new_mean = new_mean.detach().squeeze().squeeze().numpy()
        new_std = new_std.detach().squeeze().squeeze().numpy()
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

        # the acqui-optimization needs to return the best budget and the corresponding objective value
        print('\nf_best', f_best)
        print(np.shape(f_best))
        self.current_best_measurement = x_best[np.argmin(f_best)]

    def updateNP(self):
        """
        Updates the internal model with the next acquired point and its evaluation.
        """
        kw = {param: self.current_best_measurement[i]
              for i, param in enumerate(self.parameter_key)}
        param = np.array(list(kw.values()))

        # f_new is always the newest measurement for the objective function, not necessarily the best one
        f_new, stderr_f_new = self.f(param, **self.f_kwargs)    # returns the y corresponding to a test strategy

        print('np shape f_new: ', np.shape(f_new))

        self.trainNP(1, 10, new_elem_x=self.current_best_measurement, new_elem_y=f_new)
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

    def trainNP(self, epochs, batch_size, new_elem_x=None, new_elem_y=None,):

        self.NP.training = True

        print(self.x.size())
        print(self.y.size())

        # add one new budget and its corresponding function evaluation to the GP
        if new_elem_x is not None:
            self.x = torch.cat((self.x, new_elem_x), 0)
            self.y = torch.cat((self.y, new_elem_y), 0)
        training_dataset = TrainingDataset(self.x, self.y)
        self.dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
        np_trainer = NeuralProcessTrainer(self.device, self.NP, self.optimizer,
                                          num_context_range=(self.num_context, self.num_context),
                                          num_extra_target_range=(self.num_target, self.num_target),
                                          print_freq=200)
        np_trainer.train(self.dataloader, epochs)

    def NP_predict(self, xnew):

        self.NP.training = False

        # tensor needs shape (batch_size, num_samples, function_dim), function_dim is equal to the number of sentinels
        target_budget_tensor = torch.tensor(xnew).float().unsqueeze(0).unsqueeze(0)
        print('shape target budget', target_budget_tensor.shape)

        for batch in self.dataloader:
            break
        x, y = batch
        x_context, y_context, _, _ = context_target_split(x[0:1], y[0:1],
                                                          self.num_context,
                                                          self.num_target)

        p_y_pred = self.NP(x_context, y_context, target_budget_tensor)
        mu = p_y_pred.loc.detach()
        sigma = p_y_pred.scale.detach()
        return mu, sigma

    def run(self, max_iter=10, init_evals=3):
        """
        Runs the Bayesian Optimization procedure.
        @param prior_y:
        @param prior_stderr:
        @param max_iter: maximum number of iterations for GPGO
        @param init_evals:  number of random samples for fitting the GP
        @param prior: list of test strategies
        @param use_prior: boolean to use the prior
        """

        self.trainNP(epochs=30, batch_size=2)

        # get the best y and corresponding stderr
        i = np.argmax(self.prior_y)
        self.tau = np.round(self.prior_y[i], decimals=8)
        tau_stderr = self.prior_stderr[i]

        self.history.append(self.tau)
        self.stderr[self.tau] = tau_stderr

        # print(f'Running GPGO for {max_iter} iterations.')
        for _ in tqdm(range(max_iter), leave=False):

            self._optimizeAcq()
            self.updateNP()
            time_optim = time.time() - self.time_start
            self.time_for_optimization.append(time_optim/60)
