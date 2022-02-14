import numpy as np
from ax.modelbridge.strategies.alebo import ALEBOStrategy
from ax.modelbridge.strategies.rembo import HeSBOStrategy, REMBOStrategy
from ax.service.managed_loop import optimize
from matplotlib import pyplot as plt


def bo_alebo(n_nodes, total_trials,
             n_simulations, indices, eval_function, statistic, total_budget, path_plot):
    """

    @param path_plot:
    @param total_trials:
    @param total_budget: object
    @param statistic:
    @param n_nodes:
    @param eval_function: function object,
    @param n_simulations: int, number of times the objective function will run a simulation for averaging the output
    @param indices: list, indices of x in the higher dimensional x
    @return:
    """

    # the objective function has to be defined here, because it depends on a number of parameters defining the
    # trading network graph. However, AX does not allow objective functions with more than one argument.
    # Therefore the variables inside of the objective functions are treated as global inside the namespace of the bo_alebo function
    def alebo_objective_function(x):
        """
        Maps a lower dimensional x to their corresponding indices in the input vector of the given objective function.
        The input vector x_true is zero at every index except at the indices of x.

        The sum of all values of x_true (or x) is checked to be smaller or equal to the total budget.
        If this constraint is violated the function return 1e10, otherwise the output of the eva function
        (the evaluate function of the SI-model) for n_simulations is returned.

        @param x: numpy array,

        @return: float, objective function value at x
        """
        x = np.array(list(x.values()))

        assert np.shape(x) == np.shape(indices)

        # create a dummy vector to be filled with the values of x at the appropriate indices
        x_true = np.zeros(n_nodes)
        for i, xi in zip(indices, x):
            x_true[i] = xi

        if 0 < x_true.sum() <= total_budget:
            return eval_function(x_true, n_simulations=n_simulations, statistic=statistic)
        else:
            # TODO change to numpy.NaN. CMA-ES handles that as explicit rejection of x
            return 1e10  # * x.sum(x)

    # define parameter space and bounds for the objective functin
    parameters = [
        {"name": f"x{i}", "type": "range", "bounds": [0.0, float(total_budget)], "value_type": "float"}
        for i in range(len(indices))]

    # # ALEBO/REMBO/HeSBO do not take constraints
    # constraints = ' + '.join([f'x{i}' for i in range(n_nodes)])
    # constraints = [constraints + f' <= {float(n_nodes)}']
    # # print('constraints', constraints)

    # TODO maybe switch to a non-REMBO strategy to allow explicit constraints
    # alebo_strategy = ALEBOStrategy(D=n_nodes, d=4, init_size=5)
    strategy2 = REMBOStrategy(D=n_nodes, d=6, init_per_proj=2)
    # strategy3 = HeSBOStrategy(D=n_nodes, d=6, init_per_proj=10, name=f"HeSBO, d=d")

    best_parameters, values, experiment, model = optimize(
        parameters=parameters,
        experiment_name="alebo_si",
        objective_name="objective",
        evaluation_function=alebo_objective_function,
        minimize=True,
        total_trials=total_trials,
        generation_strategy=strategy2,
        # parameter_constraints=constraints
        )

    # Extract out the objective values at each iteration and make a plot
    objectives = np.array([trial.objective_mean for trial in experiment.trials.values()])

    print(objectives)
    print(np.shape(objectives))

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax.grid(alpha=0.2)
    ax.plot(range(1, total_trials+1), np.minimum.accumulate(objectives))
    # adds a line at the specified y, indicates the minimum of the branim function
    # ax.axhline(y=branin.fmin, ls='--', c='k')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Best objective found')
    plt.savefig(path_plot)
    plt.show()

    return best_parameters, values, experiment, model


if __name__ == '__main__':
    # def branin_evaluation_function(parameterization):
    #     # Evaluates Branin on the first two parameters of the parameterization.
    #     # Other parameters are unused.
    #     x = np.array([parameterization["x0"], parameterization["x1"]])
    #     return {"objective": (branin(x), 0.0)}
    print('bo_alebo')
