import numpy as np
from ax.modelbridge.strategies.alebo import ALEBOStrategy
from ax.modelbridge.strategies.rembo import HeSBOStrategy, REMBOStrategy
from ax.service.managed_loop import optimize
from matplotlib import pyplot as plt

# def branin_evaluation_function(parameterization):
#     # Evaluates Branin on the first two parameters of the parameterization.
#     # Other parameters are unused.
#     x = np.array([parameterization["x0"], parameterization["x1"]])
#     return {"objective": (branin(x), 0.0)}


def bo_alebo(objective_function, n_nodes):
    """

    @param objective_function:
    @param n_nodes:
    @return:
    """
    parameters = [
        {"name": f"x{i}", "type": "range", "bounds": [0.0, float(n_nodes)], "value_type": "float"}
        for i in range(n_nodes)]

    # ALEBO/REMBO/HeSBO do not take constraints
    constraints = ' + '.join([f'x{i}' for i in range(n_nodes)])
    constraints = [constraints + f' <= {float(n_nodes)}']

    # Setup the ALEBO optimization strategy
    # We must specify the ambient dimensionality (determined by the problem - here 100),
    # the embedding dimensionality, and the number of initial random points. As
    # discussed in the paper, the embedding dimensionality should be larger than the
    # true subspace dimensionality. Here we use 4, since the true dimensionality is 2;
    # see the paper for more discussion on this point.

    alebo_strategy = ALEBOStrategy(D=n_nodes, d=4, init_size=5)
    strategy2 = REMBOStrategy(D=n_nodes, d=6, init_per_proj=2)
    strategy3 = HeSBOStrategy(D=n_nodes, d=6, init_per_proj=10, name=f"HeSBO, d=d")

    # Run the optimization loop with that strategy
    # This will take about 30 mins to run

    best_parameters, values, experiment, model = optimize(
        parameters=parameters,
        experiment_name="alebo_si",
        objective_name="objective",
        evaluation_function=objective_function,
        minimize=True,
        total_trials=30,
        generation_strategy=strategy2,
        # parameter_constraints=constraints
    )

    # Extract out the objective values at each iteration and make a plot
    objectives = np.array([trial.objective_mean for trial in experiment.trials.values()])

    print('objectives', objectives)
    print(type(objectives), np.shape(objectives))

    print('np.minimum.accumulate(objectives)', np.minimum.accumulate(objectives))

    print('best_parameters', best_parameters)

    print('values', values, np.shape(values), type(model))
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax.grid(alpha=0.2)
    ax.plot(range(1, 31), np.minimum.accumulate(objectives))
    # ax.axhline(y=branin.fmin, ls='--', c='k') # adds a line at the specified y, indicates the minimum of the branim function
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Best objective found')
    plt.show()

    return best_parameters, values, experiment, model


if __name__ == '__main__':

    print('bo_alebo')