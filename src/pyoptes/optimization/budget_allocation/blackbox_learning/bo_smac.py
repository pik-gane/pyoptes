import os
import time
import numpy as np
import pylab as plt
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

from smac.configspace import ConfigurationSpace
from smac.facade.smac_bb_facade import SMAC4BB
from smac.optimizer.acquisition import EI

from smac.scenario.scenario import Scenario


def bo_smac(initial_population, max_iterations, n_simulations, node_indices, n_nodes, eval_function,
            statistic, total_budget, node_mapping_func, path_experiment, parallel, cpu_count, log_level=10):
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
    LOG_INTERVAL = int(max_iterations/100*10)

    # T_START = time.time()

    # SMAC is not able to pass additional arguments to function
    def smac_objective_function(x):
        """

        @param x:
        @return:
        """
        assert np.shape(x) == np.shape(NODE_INDICES)

        # create progress messages every llog_interval iteration
        if LOG_ITERATOR[0] != 0 and LOG_ITERATOR[0] % LOG_INTERVAL == 0:
            print(f'\nIteration: {LOG_ITERATOR[0]}/{MAX_ITERATIONS}. '
                  f'Minutes elapsed since start: {(time.time()-T_START)/60}\n')
        LOG_ITERATOR[0] = LOG_ITERATOR[0]+1

        # convert the smac dict to a numpy array
        x = np.array(list(x.values()))

        x_true = node_mapping_func(x, N_NODES, NODE_INDICES)

        # compute a penalty for violating sum constraint
        x_true_normed = x_true/x_true.sum() #total_budget * np.exp(x) / sum(np.exp(x))
        x_true_scaled = x_true_normed*total_budget

        sqr_diff_x = (x_true.sum()-x_true_scaled.sum())**2 # ~200k

        return EVAL_FUNCTION(x_true_scaled, n_simulations=N_SIMULATIONS, statistic=STATISTIC,
                             parallel=PARALLEL, num_cpu_cores=CPU_COUNT) + sqr_diff_x*10

    # Build Configuration Space which defines all parameters and their ranges
    cs = ConfigurationSpace()
    parameter_space = [UniformFloatHyperparameter(f'x{i}', 0.0, total_budget, default_value=xi)
                       for i, xi in enumerate(initial_population)]
    cs.add_hyperparameters(parameter_space)

    scenario = Scenario({"run_obj": "quality",
                         "runcount-limit": max_iterations,  # max. number of function evaluations
                         "cs": cs,  # configuration space
                         "deterministic": "True",
                         "output_dir": 'smac3_output'})
                         # "shared_model": True,
                         # "input_psmac_dirs": 'smac3_output2'})

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

    # TODO add info to the plot whether the current test strategy violates the sum constraint
    plt.plot(range(len(ys)), ys)
    plt.title('SMAC')
    plt.xlabel('Iteration')
    plt.ylabel('SI-model output')
    plt.savefig(os.path.join(path_experiment, 'smac_plot.png'))

    return optimum


if __name__ == "__main__":
    print('success')
