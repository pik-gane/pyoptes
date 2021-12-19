"""
Simple test to illustrate how the target function could be used.
"""

import numpy as np
import pylab as plt
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

print("\nOne evaluation at random x:", y)

n_trials = 1000

# evaluate f a number of times at the same input:
ys = np.array([f.evaluate(x) for it in range(n_trials)])
logys = np.log(ys)
print("Mean and std.dev. of", n_trials, "evaluations at the same random x:", ys.mean(), ys.std())
print("Mean and std.dev. of the log of", n_trials, "evaluations at that x:", logys.mean(), logys.std())


# do the same for an x that is based on the total number of incoming transmissions per node:

target_list = f.model.transmissions_array[:, 3]
values, counts = np.unique(target_list, return_counts=True)
weights = np.zeros(n_inputs)
weights[values] = counts
shares = weights / weights.sum()
total_budget = 1.0 * n_inputs

x2 = shares * total_budget
ys2 = np.array([f.evaluate(x2) for it in range(n_trials)])
logys2 = np.log(ys2)
print("\nMean and std.dev. of", n_trials, "evaluations at a network-based x:", ys2.mean(), ys2.std())
print("Mean and std.dev. of the log of", n_trials, "evaluations at that x:", logys2.mean(), logys2.std())

plt.hist(ys)
plt.hist(ys2)
plt.title("histogram of f(x) for two different fixed values of x")
plt.show()

plt.hist(logys)
plt.hist(logys2)
plt.title("histogram of log(f(x)) for two different fixed values of x")
plt.show()
