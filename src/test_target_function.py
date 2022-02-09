"""
Simple test to illustrate how the target function could be used.
Here focussing on the AVERAGE no. of infected animals
hence using a target function based on a SINGLE simulation
so that the optimizer can later decide how often to evaluate it. 
"""

import numpy as np
from scipy.stats import gaussian_kde as kde
import pylab as plt
from pyoptes import set_seed
from pyoptes.optimization.budget_allocation import target_function as f

print(f.__doc__)

# set some seed to get reproducible results:
set_seed(2)

print("Preparing the target function for a random but fixed transmissions network")

# at the beginning, call prepare() once:
f.prepare(
    n_nodes=120,  # instead of 60000, since this should suffice in the beginning
    capacity_distribution=np.random.lognormal,  # this is more realistic than a uniform distribution
    delta_t_symptoms=60  # instead of 30, since this gave a clearer picture in Sara's simulations
    )
n_inputs = f.get_n_inputs()
print("n_inputs (=number of network nodes):", n_inputs)

total_budget = 1.0 * n_inputs  # i.e., on average, nodes will do one test per year

# evaluate f once at a random input:
weights = np.random.rand(n_inputs)
shares = weights / weights.sum()
x = shares * total_budget  
y = f.evaluate(x)

print("\nOne evaluation at random x:", y)

n_trials = 1000

# evaluate f a number of times at the same input:
ys = np.array([f.evaluate(x) for it in range(n_trials)])
logys = np.log(ys)
print("Mean, std.dev. and 95th percentile of", n_trials, "evaluations at the same random x:", ys.mean(), ys.std(), np.quantile(ys,0.95))
print("Mean, std.dev. and 95th percentile of the log of", n_trials, "evaluations at that x:", logys.mean(), logys.std(), np.quantile(logys,0.95))


# do the same for an x that is based on the total capacity of a node:

weights = f.capacities
shares = weights / weights.sum()

x2 = shares * total_budget
ys2 = np.array([f.evaluate(x2) for it in range(n_trials)])
logys2 = np.log(ys2)
print("\nMean, std.dev. and 95th percentile of", n_trials, "evaluations at a capacity-based x:", ys2.mean(), ys2.std(), np.quantile(ys2,0.95))
print("Mean, std.dev. and 95th percentile of the log of", n_trials, "evaluations at that x:", logys2.mean(), logys2.std(), np.quantile(logys2,0.95))


# do the same for an x that is based on the total number of incoming transmissions per node:

target_list = f.model.transmissions_array[:, 3]
values, counts = np.unique(target_list, return_counts=True)
weights = np.zeros(n_inputs)
weights[values] = counts
shares = weights / weights.sum()
total_budget = 1.0 * n_inputs

x3 = shares * total_budget
ys3 = np.array([f.evaluate(x3) for it in range(n_trials)])
logys3 = np.log(ys3)
print("\nMean, std.dev. and 95th percentile of", n_trials, "evaluations at a transmissions-based x:", ys3.mean(), ys3.std(), np.quantile(ys3,0.95))
print("Mean, std.dev. and 95th percentile of the log of", n_trials, "evaluations at that x:", logys3.mean(), logys3.std(), np.quantile(logys3,0.95))


# do the same for an x that is based on the static network's node degrees:

weights = np.zeros(n_inputs)
for n, d in f.network.degree():
    weights[n] = d
shares = weights / weights.sum()

x4 = shares * total_budget
ys4 = np.array([f.evaluate(x4) for it in range(n_trials)])
logys4 = np.log(ys4)
print("\nMean, std.dev. and 95th percentile of", n_trials, "evaluations at a degree-based x:", ys4.mean(), ys4.std(), np.quantile(ys4,0.95))
print("Mean, std.dev. and 95th percentile of the log of", n_trials, "evaluations at that x:", logys4.mean(), logys4.std(), np.quantile(logys4,0.95))



xs = np.linspace(ys3.min(), ys.max())
plt.plot(xs, kde(ys)(xs), label="random x")
plt.plot(xs, kde(ys2)(xs), alpha=0.5, label="capacity-based x")
plt.plot(xs, kde(ys4)(xs), alpha=0.5, label="degree-based x")
plt.plot(xs, kde(ys3)(xs), alpha=0.5, label="transmissions-based x")
plt.legend()
plt.title("distribution of f(x) for different fixed inputs x")
plt.show()

xs = np.linspace(logys3.min(), logys.max())
plt.plot(xs, kde(logys)(xs), label="random x")
plt.plot(xs, kde(logys2)(xs), alpha=0.5, label="capacity-based x")
plt.plot(xs, kde(logys4)(xs), alpha=0.5, label="degree-based x")
plt.plot(xs, kde(logys3)(xs), alpha=0.5, label="transmissions-based x")
plt.legend()
plt.title("distribution of log(f(x)) for different fixed inputs x")
plt.show()
