# Define a function to be optimized.
# Here we use a simple synthetic function with a d=2 true linear embedding, in
# a D=100 ambient space.
import numpy as np
from ax.utils.measurement.synthetic_functions import branin
from ax.modelbridge.strategies.alebo import ALEBOStrategy
from ax.modelbridge.strategies.rembo import HeSBOStrategy, REMBOStrategy
from ax.service.managed_loop import optimize
from matplotlib import pyplot as plt

from pyoptes import set_seed
from pyoptes.optimization.budget_allocation import target_function as f


# def branin_evaluation_function(parameterization):
#     # Evaluates Branin on the first two parameters of the parameterization.
#     # Other parameters are unused.
#     x = np.array([parameterization["x0"], parameterization["x1"]])
#     return {"objective": (branin(x), 0.0)}


if __name__ == '__main__':

    # set some seed to get reproducible results:
    set_seed(1)

    n_nodes = 120
    nn_simulations = 2
    print('\nn_simulations outside evaluate', nn_simulations,'\n')
    # at the beginning, call prepare() once:
    f.prepare(n_nodes=n_nodes)

    # weird hack, cma-es only takes function objects and the default value n_simula
    def evaluate(x):#, nn_simulations=nn_simulations):
        x = np.array(list(x.values()))
        print(x[:3])
        # print(type(x), np.shape(x))
        # print('n_simulations inside evaluate', nn_simulations)
        return f.evaluate(x)#, n_simulations=nn_simulations), 0.0)}

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
        evaluation_function=evaluate,
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

    print('values',values, np.shape(values), type(model))
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax.grid(alpha=0.2)
    ax.plot(range(1, 31), np.minimum.accumulate(objectives))
    # ax.axhline(y=branin.fmin, ls='--', c='k') # adds a line at the specified y, indicates the minimum of the branim function
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Best objective found')
    plt.show()