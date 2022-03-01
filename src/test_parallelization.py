"""
Test comparing the performance (time and simulation output) of the target function with and without parallelization.
"""

from time import time
import numpy as np
import pylab as plt
from pyoptes import set_seed
from pyoptes.optimization.budget_allocation import target_function as fp
from tqdm import tqdm

set_seed(2)

n_simulations = 10000
n_nodes = 120

fp.prepare(
    n_nodes=n_nodes,  # instead of 60000, since this should suffice in the beginning
    capacity_distribution=np.random.lognormal,  # this is more realistic than a uniform distribution
    delta_t_symptoms=60  # instead of 30, since this gave a clearer picture in Sara's simulations
    )

total_budget = 1.0 * n_nodes

# init test strategy
weights = np.random.rand(n_nodes)
shares = weights / weights.sum()
x = shares * total_budget

a = time()
y = fp.evaluate(x, n_simulations=n_simulations, parallel=False)
b = time()-a
print('Non-parallel simulation')
print(f'Time for {n_simulations} simulations of: {b}')
print(f'y: {y}')


a = time()
y = fp.evaluate(x, n_simulations=n_simulations)
b = time()-a
print('Parallel simulation')
print(f'Time for {n_simulations} simulations of: {b}')
print(f'y: {y}')

n = [1, 100, 1000, 10000, 100000, 1000000]

tl = []
yl = []

tlp = []
ylp = []
for s in tqdm(n):
    a = time()
    yl.append(fp.evaluate(x, n_simulations=s, parallel=False))
    tl.append(time() - a)

    a = time()
    ylp.append(fp.evaluate(x, n_simulations=s))
    tlp.append(time() - a)

plt.plot(n, yl, label='y non-parallel')
plt.plot(n, ylp, label='y parallel')
plt.title('Simulation output for 1, 100, 1000, 10000, 1000000 evaluations')
plt.legend()
plt.show()

plt.plot(n, tl, label='time non-parallel')
plt.plot(n, tlp, label='time parallel')
plt.title('Evaluation time for  for 1, 100, 1000, 10000, 1000000 evaluations')
plt.legend()
plt.show()
