import os
import numpy as np
import pylab as plt
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

from smac.configspace import ConfigurationSpace
from smac.facade.smac_bb_facade import SMAC4BB
from smac.optimizer.acquisition import EI

from smac.scenario.scenario import Scenario


def bo_smac(initial_population, max_iterations, n_simulations, node_indices, n_nodes, eval_function,
            statistic, total_budget, node_mapping_func, path_experiment):
    """


    Adapted from SMAC-tutorial: https://automl.github.io/SMAC3/master/pages/examples/python/plot_synthetic_function.html
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

    # SMAC is not able to pass additional arguments to function
    def smac_objective_function(x):
        """

        @param x:
        @return:
        """
        assert np.shape(x) == np.shape(NODE_INDICES)
        # convert the smac dict to a numpy array
        x = np.array(list(x.values()))

        x_true = node_mapping_func(x, N_NODES, NODE_INDICES)

        if 0 < x_true.sum() <= TOTAL_BUDGET:
            return EVAL_FUNCTION(x_true, n_simulations=N_SIMULATIONS, statistic=STATISTIC)
        else:
            return 1e10  # * x.sum(x) # np.NaN doesn't work

    # Build Configuration Space which defines all parameters and their ranges
    cs = ConfigurationSpace()
    parameter_space = [UniformFloatHyperparameter(f'x{i}', 0.0, total_budget, default_value=xi)
                       for i, xi in enumerate(initial_population)]
    cs.add_hyperparameters(parameter_space)

    scenario = Scenario({"run_obj": "quality",
                         "runcount-limit": max_iterations,  # max. number of function evaluations
                         "cs": cs,  # configuration space
                         "deterministic": "True",
                         "output_dir": 'smac3_output',
                         "verbose_level": "18"})

    # Optimize, using a SMAC-object
    smac = SMAC4BB(scenario=scenario,
                   model_type='gp',
                   rng=np.random.RandomState(42),
                   acquisition_function=EI,  # or others like PI, LCB as acquisition functions
                   tae_runner=smac_objective_function)
    optimum = smac.optimize()

    rh = smac.get_runhistory()
    ys = []
    for (config_id, instance_id, seed, budget), (cost, time, status, starttime, endtime, additional_info) in rh.data.items():
        config = rh.ids_config[config_id]
        ys.append(smac_objective_function(config))

    plt.plot(range(len(ys)), ys)
    plt.title('SMAC')
    plt.xlabel('Iteration')
    plt.ylabel('SI-model output')
    plt.savefig(os.path.join(path_experiment, 'smac_plot.png'))

    return optimum


if __name__ == "__main__":
    print('success')
