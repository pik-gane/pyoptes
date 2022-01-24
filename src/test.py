import numpy as np
from pyoptes import set_seed
from pyoptes.optimization.budget_allocation import target_function as f
from tqdm import tqdm



if __name__ == '__main__':

    # set some seed to get reproducible results:
    set_seed(1)

    n_nodes = 120
    n_simulations = 1000
    # at the beginning, call prepare() once:
    f.prepare(n_nodes=n_nodes)

    n_inputs = f.get_n_inputs()
    print('n_inputs', n_inputs)

    total_budget = 1.0 * n_inputs  # i.e., on average, nodes will do one test per year
    print('total_budget', total_budget)
    # evaluate f once at a random input:
    weights = np.random.rand(n_inputs)
    # print('weights', weights)
    print('weights.sum()', weights.sum())
    shares = weights / weights.sum()
    # print('shares', shares)
    x = shares * total_budget
    # print('x', x)
    print('x.sum()', x.sum())

    # take 120 biggest nodes

    # take 120 most connected nodes