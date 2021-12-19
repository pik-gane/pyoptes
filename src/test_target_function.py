"""
Simple test to illustrate how the target function could be used.
"""

import numpy as np
from pyoptes import set_seed
from pyoptes.optimization.budget_allocation import target_function as f

print(f.__doc__)

# set some seed to get reproducible results:
set_seed(1)

# at the beginning, call prepare() once:
f.prepare()
n_inputs = f.get_n_inputs()
print("n_inputs:", n_inputs)

# evaluate f once at a random input:
x = 2 * np.random.rand(n_inputs)  # i.e., on average, nodes will do one test per year
y = f.evaluate(x)

print("One evaluation at random x:", y)

# evaluate f a number of times at the same input:
mean_y = np.mean([f.evaluate(x) for it in range(10)])
print("Mean of ten evaluations at the same x:", mean_y)

