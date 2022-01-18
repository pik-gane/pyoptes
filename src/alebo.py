# Define a function to be optimized.
# Here we use a simple synthetic function with a d=2 true linear embedding, in
# a D=100 ambient space.
import numpy as np
from ax.utils.measurement.synthetic_functions import branin
from ax.modelbridge.strategies.alebo import ALEBOStrategy
from ax.service.managed_loop import optimize
from matplotlib import pyplot as plt


def branin_evaluation_function(parameterization):
    # Evaluates Branin on the first two parameters of the parameterization.
    # Other parameters are unused.
    x = np.array([parameterization["x0"], parameterization["x1"]])
    return {"objective": (branin(x), 0.0)}


if __name__=='__main__':
    # Define the parameters in the format expected by Ax.
    # Here we define a D=100 search space by augmenting the real Branin parameters
    # with 98 unused parameters.

    parameters = [
        {"name": "x0", "type": "range", "bounds": [-5.0, 10.0], "value_type": "float"},
        {"name": "x1", "type": "range", "bounds": [0.0, 15.0], "value_type": "float"},
    ]
    parameters.extend([
        {"name": f"x{i}", "type": "range", "bounds": [-5.0, 10.0], "value_type": "float"}
        for i in range(2, 100)
    ])
    # Setup the ALEBO optimization strategy
    # We must specify the ambient dimensionality (determined by the problem - here 100),
    # the embedding dimensionality, and the number of initial random points. As
    # discussed in the paper, the embedding dimensionality should be larger than the
    # true subspace dimensionality. Here we use 4, since the true dimensionality is 2;
    # see the paper for more discussion on this point.

    alebo_strategy = ALEBOStrategy(D=100, d=4, init_size=5)

    # Run the optimization loop with that strategy
    # This will take about 30 mins to run

    best_parameters, values, experiment, model = optimize(
        parameters=parameters,
        experiment_name="branin_100",
        objective_name="objective",
        evaluation_function=branin_evaluation_function,
        minimize=True,
        total_trials=30,
        generation_strategy=alebo_strategy,
    )

    # Extract out the objective values at each iteration and make a plot
    objectives = np.array([trial.objective_mean for trial in experiment.trials.values()])

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax.grid(alpha=0.2)
    ax.plot(range(1, 31), np.minimum.accumulate(objectives))
    ax.axhline(y=branin.fmin, ls='--', c='k')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Best objective found');