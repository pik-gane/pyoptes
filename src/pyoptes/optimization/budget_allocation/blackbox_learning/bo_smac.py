import numpy as np

import numpy as np
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

from smac.configspace import ConfigurationSpace
from smac.facade.smac_bb_facade import SMAC4BB
from smac.optimizer.acquisition import EI

from smac.scenario.scenario import Scenario


def bo_smac(initial_population, max_iterations, n_simulations, node_indices, n_nodes, eval_function,
            statistic, total_budget):
    """


    Adapted from SMAC-tutorial: https://automl.github.io/SMAC3/master/pages/examples/python/synthetic_function.html
    @param initial_population:
    @param max_iterations:
    @param n_simulations:
    @param node_indices:
    @param n_nodes:
    @param eval_function:
    @param statistic:
    @param total_budget:
    @return:
    """
    EVAL_FUNCTION = eval_function
    N_SIMULATIONS = n_simulations
    NODE_INDICES = node_indices
    STATISTIC = statistic
    TOTAL_BUDGET = total_budget
    N_NODES = n_nodes

    def smac_objective_function(x):
        """

        @param x:
        @return:
        """
        assert np.shape(x) == np.shape(NODE_INDICES)
        # convert the smac dict to a numpy array
        x = np.array(list(x.values()))

        # create a dummy vector to be filled with the values of x at the appropriate indices
        x_true = np.zeros(N_NODES)
        for i, xi in zip(NODE_INDICES, x):
            x_true[i] = xi

        if 0 < x_true.sum() <= TOTAL_BUDGET:
            return EVAL_FUNCTION(x_true, n_simulations=N_SIMULATIONS, statistic=STATISTIC)
        else:
            # TODO change to numpy.NaN. CMA-ES handles that as explicit rejection of x
            return 1e10  # * x.sum(x)

    # Build Configuration Space which defines all parameters and their ranges
    cs = ConfigurationSpace()
    parameter_space = [UniformFloatHyperparameter(f'x{i}', 0.0, total_budget, default_value=xi)
                       for i, xi in enumerate(initial_population)]
    cs.add_hyperparameters(parameter_space)

    scenario = Scenario({"run_obj": "quality",
                         "runcount-limit": max_iterations,  # max. number of function evaluations
                         "cs": cs,  # configuration space
                         "deterministic": "false"})

    # Optimize, using a SMAC-object
    smac = SMAC4BB(scenario=scenario,
                   model_type='gp',
                   rng=np.random.RandomState(42),
                   acquisition_function=EI,  # or others like PI, LCB as acquisition functions
                   tae_runner=smac_objective_function)

    return smac.optimize()


# TODO maybe enforce correct types of params ? To prevent floats where ints are expected
def smac_objective_function(x, n_simulations, node_indices, n_nodes, eval_function, statistic, total_budget):
    """
    An optimizeable objective function.
    Maps a lower dimensional x to their corresponding indices in the input vector of the given objective function.
    The input vector x_true is zero at every index except at the indices of x.

    The sum of all values of x_true (or x) is checked to be smaller or equal to the total budget.
    If this constraint is violated the function return 1e10, otherwise the output of the eva function
    (the evaluate function of the SI-model) for n_simulations is returned.

    @param total_budget: float,
    @param statistic: function object,
    @param x: numpy array,
    @param eval_function: function object,
    @param n_simulations: int, number of times the objective function will run a simulation for averaging the output
    @param node_indices: list, indices of x in the higher dimensional x
    @param n_nodes: int, dimension of the input of the objective function
    @return: float, objective function value at x
    """
    assert np.shape(x) == np.shape(node_indices)
    # create a dummy vector to be filled with the values of x at the appropriate indices
    x_true = np.zeros(n_nodes)
    for i, xi in zip(node_indices, x):
        x_true[i] = xi
    if 0 < x_true.sum() <= total_budget:
        return eval_function(x_true, n_simulations=n_simulations, statistic=statistic)
    else:
        return np.NaN#1e10     # * x.sum(x)


if __name__ == "__main__":
    print('success')
