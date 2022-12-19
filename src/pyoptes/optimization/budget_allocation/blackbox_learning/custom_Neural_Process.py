'''
Adaption of the GPGO-class from pyGPGO.
Extended use a neural process as the surrogate function.
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
from torch.distributions import Normal


class NP:
    def __init__(self, acquisition, f, parameter_dict, prior_x, prior_y, prior_stderr,
                 epochs: int, batch_size: int,
                 n_jobs: int = 15, f_kwargs={},
                 r_dim: int = 50,  # Dimension of representation of context points
                 z_dim: int = 50,  # Dimension of sampled latent variable
                 h_dim: int = 50,  # Dimension of hidden layers in encoder and decoder
                 num_context: int = 3,  # num_context + num_target has to be lower than num_samples
                 num_target: int = 3,
                 z_sample_size: int = 10,
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

        # neural process hyper-parameters
        # dimension of the objective function, equal to the number of sentinels
        self.x_dim = len(f_kwargs['node_indices'])

        self.num_context = num_context
        self.num_target = num_target

        self.NP = NeuralProcess(self.x_dim, r_dim=r_dim, z_dim=z_dim, h_dim=h_dim, y_dim=1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = torch.optim.Adam(self.NP.parameters(), lr=3e-4)

        self.epochs = epochs
        self.batch_size = batch_size

        self.z_sample_size = z_sample_size

        # ----

        self.prior_x = prior_x
        self.prior_y = -1*np.array(prior_y) # TODO why is this necessary?
        self.prior_stderr = prior_stderr

        self.x = torch.tensor(np.array(prior_x)).unsqueeze(0).float()
        self.y = torch.tensor(np.array(prior_y)).unsqueeze(1).unsqueeze(0).float()

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

        # history contains the best function evaluations in every iteration
        self.history = []
        # dictionary containing the stderr of each objective function call (keys are function values)
        self.stderr = {}
        # contains the function evaluations (for the proposed budget) in every iteration, not necessarily the best value
        self.surrogate_y = [] # TODO maybe rename ?

        self.current_best_budget = {} # TODO needs better name

        self.time_for_optimization = []
        self.time_acquisition_optimization = []
        self.time_update_surrogate = []

        # keyword arguments for the objective function
        self.f_kwargs = f_kwargs

        # flags for saving test strategies
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

        new_mean = new_mean.detach().reshape(-1).numpy()
        new_std = new_std.detach().reshape(-1).numpy()
        return -self.A.eval(self.tau, new_mean, new_std)

    def _optimizeAcq(self, method='L-BFGS-B', n_start=100, ):
        """
        Optimizes the acquisition function using a multistart approach.

        Parameters
        ----------
        method: str. Default 'L-BFGS-B'.
            Any `scipy.optimize` method that admits bounds and gradients. Default is 'L-BFGS-B'.
        n_start: int.
            Number of starting points for the optimization procedure. Default is 100.

        """
        time_acqui = time.time()

        # sample starting points and append the current best budget
        start_points_list = [self._sampleParam() for _ in range(n_start)]
        start_points_list.append(self.current_best_budget[self.tau])
        start_points_array = np.array(start_points_list)

        # save the starting points for visualization
        if self.save_test_strategies:
            np.save(os.path.join(self.save_test_strategies_path, f'test_strategy_{self.n}'), start_points_array)
            self.n += 1

        # evaluate the surrogate model at the starting points. Either sequentially or in parallel
        x_best = np.empty((n_start, len(self.parameter_key)))
        f_best = np.empty((n_start,))
        if self.n_jobs == 1:
            for index, start_point in enumerate(start_points_array):
                res = minimize(self._acqWrapper, x0=start_point, method=method,
                               bounds=self.parameter_range)
                x_best[index], f_best[index] = res.x, np.atleast_1d(res.fun)[0]
        else:
            opt = Parallel(n_jobs=self.n_jobs)(delayed(minimize)(self._acqWrapper,
                                                                 x0=start_point,
                                                                 method=method,
                                                                 bounds=self.parameter_range) for start_point in
                                               start_points_array)
            x_best = np.array([res.x for res in opt])
            f_best = np.array([np.atleast_1d(res.fun)[0] for res in opt])

        # the acqui-optimization needs to return the best budget and the corresponding objective value
        self.current_best_measurement = x_best[np.argmin(f_best)]

        # save time for acquisition function optimization
        time_acqui = time.time() - time_acqui
        self.time_acquisition_optimization.append(time_acqui/60)

    def getResult(self):
        """
        Prints best result in the Bayesian Optimization procedure.

        Returns
        -------
        float
            Best function evaluation.

        """
        argtau = np.max(self.surrogate_y)
        best_budget = self.current_best_budget[argtau]

        return best_budget, self.tau

    def updateNP(self):
        """
        Updates the internal model with the next acquired point and its evaluation.
        """
        time_surrogate = time.time()

        kw = {param: self.current_best_measurement[i]
              for i, param in enumerate(self.parameter_key)}
        param = np.array(list(kw.values()))

        # f_new is always the newest measurement for the objective function, not necessarily the best one
        f_new, stderr_f_new = self.f(param, **self.f_kwargs)    # returns the y corresponding to a test strategy

        self.trainNP(epochs=self.epochs, batch_size=self.batch_size,
                     new_elem_x=self.current_best_measurement, new_elem_y=f_new)

        # add new stderr to the stderr dictionary. This ensures that there is always a stderr for each measurement
        f_new = np.round(f_new, decimals=8)
        self.surrogate_y.append(f_new)
        self.stderr[f_new] = stderr_f_new
        self.current_best_budget[f_new] = param

        # TODO NP class does not keep a list of max Ys,
        # get the current optimum of the GP
        self.tau = np.max(self.surrogate_y) # self.GP "saves" the y from the objective f,
        self.tau = np.round(self.tau, decimals=8)
        self.history.append(self.tau)

        # save time for surrogate model update
        time_surrogate = time.time() - time_surrogate
        self.time_update_surrogate.append(time_surrogate/60)

    def trainNP(self, epochs, batch_size, new_elem_x=None, new_elem_y=None,):

        # train the neural process on the available data. Either the initial data or the new data
        self.NP.training = True

        # TODO there possibly needs to be a -1
        # add one new budget and its corresponding function evaluation to the GP
        if new_elem_x is not None:
            # reshape data to fit the neural process
            new_elem_x = torch.tensor(new_elem_x).reshape((1, 1, self.x_dim)).float()
            new_elem_y = torch.tensor(new_elem_y).reshape((1, 1, 1)).float()
            # TODO something is wrong with the shapes and cat here
            # print('np shape new_elem_x: ', np.shape(new_elem_x), type(new_elem_x), new_elem_x)
            # print('np shape new_elem_y: ', np.shape(new_elem_y), type(new_elem_y), new_elem_y)
            self.x = torch.cat((self.x, new_elem_x), 1)
            self.y = torch.cat((self.y, new_elem_y), 1)
        training_dataset = TrainingDataset(self.x, self.y)
        self.dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
        np_trainer = NeuralProcessTrainer(self.device, self.NP, self.optimizer,
                                          num_context_range=(self.num_context, self.num_context),
                                          num_extra_target_range=(self.num_target, self.num_target),
                                          print_freq=200)
        np_trainer.train(self.dataloader, epochs)

        # use the first element in the data as the context for the next prediction
        for batch in self.dataloader:
            break
        x, y = batch
        x_context, y_context, _, _ = context_target_split(x[0:1], y[0:1],
                                                          self.num_context,
                                                          self.num_target)

        # whenever the neural process is trained (or updated), z is updated as well
        mu_context, sigma_context = self.NP.xy_to_mu_sigma(x_context, y_context)
        # Sample from distribution based on context
        q_context = Normal(mu_context, sigma_context)
        # sample from the distribution ten times and take the mean
        self.z_sample = q_context.rsample((self.z_sample_size,)).mean(0)

    def NP_predict(self, xnew):

        self.NP.training = False

        # tensor needs shape (batch_size, num_samples, function_dim), function_dim is equal to the number of sentinels
        target_budget_tensor = torch.tensor(xnew).float().unsqueeze(0).unsqueeze(0)
        # print('shape target budget', target_budget_tensor.shape)

        p_y_pred = self.NP(x_target=target_budget_tensor, z_sample_predict=self.z_sample,
                           x_context=None, y_context=None)
        mu = p_y_pred.loc.detach()
        sigma = p_y_pred.scale.detach()
        return mu, sigma

    def run(self, max_iter=10):
        """
        Runs the Bayesian Optimization procedure.
        @param max_iter: maximum number of iterations for GPGO
        """

        self.trainNP(epochs=self.epochs, batch_size=self.batch_size)

        # get the best y and corresponding stderr and save it as self.tau
        i = np.argmax(self.prior_y)
        self.tau = np.round(self.prior_y[i], decimals=8)
        tau_stderr = self.prior_stderr[i]

        # current best budget is a dictionary with the y/tau as key and the corresponding budget as value
        self.current_best_budget[self.tau] = self.prior_x[i]
        # surrogate_y is a list of all the best y values that have been evaluated so far
        self.surrogate_y.append(self.tau)

        self.history.append(self.tau)
        # self.stderr contains the stderr for each tau and gets continuously updated
        self.stderr[self.tau] = tau_stderr

        time_start = time.time()
        for _ in tqdm(range(max_iter), leave=False):

            self._optimizeAcq()
            self.updateNP()

            time_optim = time.time() - time_start
            self.time_for_optimization.append(time_optim/60)
