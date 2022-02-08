from pyoptes import set_seed
from pyoptes.optimization.budget_allocation import target_function as f

from pyoptes.optimization.budget_allocation.blackbox_learning.bo_cma import bo_cma
from pyoptes.optimization.budget_allocation.blackbox_learning.bo_alebo import bo_alebo

import numpy as np
import argparse


def choose_high_degree_nodes(node_degrees, n):
    """
    Returns the indices of n nodes with the highest degrees.
    @param node_degrees: list, contains indices of nodes and their degree
    @param n: int, number of nodes to be returned
    @return: list of ints
    """

    nodes_sorted = sorted(node_degrees, key=lambda node_degrees: node_degrees[1], reverse=True)
    indices_highest_degree_nodes = [i[0] for i in nodes_sorted[:n]]
    return indices_highest_degree_nodes


# TODO find descriptive name for function
# TODO rename true_size_x to n_nodes ??
def of(x, n_simulations, indices, true_size_x, eval_function):
    """
    Maps a lower dimensional x to their corresponding indices in the input vector of the given  objective function.
    @param x: numpy array,
    @param eval_function: function object,
    @param n_simulations: int, number of times the objective function will run a simulation for averaging the output
    @param indices: list, indices of x in the higher dimensional x
    @param true_size_x: int, dimension of the input of the objective function
    @return: float, objective function value at x
    """
    assert np.shape(x) == np.shape(indices)
    # create a dummy vector to be filled with the values of x at the appropriate indices
    true_x = np.zeros(true_size_x)
    for i, xi in zip(indices, x):
        true_x[i] = xi

    # TODO depending on the CMA-ES options, a second check for positivity of the x values has to be added
    if x.sum() <= total_budget:
        return eval_function(true_x, n_simulations=n_simulations)
    else:
        # TODO change to numpy.NaN. CMA-ES handles that as explicit rejection of x
        return 1e10     # * x.sum(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("optimizer", choices=['cma', 'alebo'],
                        help="Choose the optimizer to run on the SI-model. Choose between CMA-ES and ALEBO")
    parser.add_argument("--n_nodes", type=int, default=120, help="Defines the number of nodes used by the SI-model. "
                                                                 "Default value is 120.")
    parser.add_argument("--n_simulations", type=int, default=1000, help=" Sets the number of runs the for the "
                                                                        "SI-model. Higher values of n_simulations lower"
                                                                        "the variance of the output of the simulation. "
                                                                        "Default value is 1000.")
    parser.add_argument("--max_iterations", type=int, default=100, help="The maximum number of iterations CMA-ES runs.")
    parser.add_argument("--size_subset", type=int, default=10, help="Set the number of nodes that are used. Has to be"
                                                                    "smaller than or equal to n_nodes")
    parser.add_argument('--cma_sigma', type=float, default=0.2, help="")
    args = parser.parse_args()

    # set some seed to get reproducible results:
    set_seed(1)
    # at the beginning, call prepare() once:
    f.prepare(n_nodes=args.n_nodes)

    total_budget = 1.0 * args.n_nodes  # i.e., on average, nodes will do one test per year
    # evaluate f once at a random input:
    weights = np.random.rand(args.size_subset)
    shares = weights / weights.sum()
    x = shares * total_budget

    def objective_function_alebo(x, total_budget=total_budget):#, nn_simulations=nn_simulations):
        x = np.array(list(x.values()))
        if x.sum() <= 120.0:
            return f.evaluate(x)#, n_simulations=nn_simulations), 0.0)}
        else:
            return 1e10     # * x.sum(x)

    # print('transmission array', f.model.transmissions_array)

    # print('capacities per node: ', f.capacities)

    # reduce the dimension of the input space
    ix = choose_high_degree_nodes(f.network.degree, args.size_subset)

    # get the first constraint, the boundaries of x_i
    bounds = [0, total_budget]

    if args.optimizer == 'cma':
        solutions = bo_cma(of, x, max_iterations=args.max_iterations,
                           n_simulations=args.n_simulations,
                           indices=ix,
                           true_size_x=args.n_nodes,
                           eval_function=f.evaluate,
                           bounds=bounds)
        print('\nCMA-ES solutions')
        for s in solutions:
            print(of(s, indices=ix,
                     true_size_x=args.n_nodes,
                     eval_function=f.evaluate,
                     n_simulations=10000))
    elif args.optimizer == 'alebo':
        best_parameters, values, experiment, model = bo_alebo(objective_function_alebo, args.n_nodes,
                                                              args.max_iterations)

        best_parameters = np.array(list(best_parameters.values()))
        print('min, max, sum: ', best_parameters.min(), best_parameters.max(), best_parameters.sum())

    else:
        print('Something went wrong with choosing the optimizer.')
