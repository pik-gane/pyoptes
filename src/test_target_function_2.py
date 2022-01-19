"""
Simple test to illustrate how the target function could be used.
Here focussing on the 95th PERCENTILE of the no. of infected animals,
hence using a target function based on MANY simulations
and reporting the std.err. of the resulting estimation of the 95th percentile.
"""

import numpy as np
from scipy.stats import gaussian_kde as kde
import pylab as plt
from pyoptes import set_seed
from pyoptes.optimization.budget_allocation import target_function as f

print(f.__doc__)

# set some seed to get reproducible results:
set_seed(1)

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
y = f.evaluate(
        x, 
        n_simulations=100, 
        statistic=lambda a: np.percentile(a, 95)  # to focus on the tail of the distribution
        )

print("\nOne evaluation at random x:", y)


evaluation_parms = { 
        'n_simulations': 100, 
        'statistic': lambda a: np.percentile(a, 95)
        }

n_trials = 1000

def stderr(a):
    return np.std(a, ddof=1) / np.sqrt(np.size(a))


# evaluate f a number of times at the same input:
ys = np.array([f.evaluate(x, **evaluation_parms) for it in range(n_trials)])
logys = np.log(ys)
print("Mean and std.err. of", n_trials, "evaluations at the same random x:", ys.mean(), stderr(ys))
print("Mean and std.err. of the log of", n_trials, "evaluations at that x:", logys.mean(), stderr(logys))


# do the same for an x that is based on the total capacity of a node:

weights = f.capacities
shares = weights / weights.sum()

x2 = shares * total_budget
ys2 = np.array([f.evaluate(x2, **evaluation_parms) for it in range(n_trials)])
logys2 = np.log(ys2)
print("\nMean and std.err. of", n_trials, "evaluations at a capacity-based x:", ys2.mean(), stderr(ys2))
print("Mean and std.err. of the log of", n_trials, "evaluations at that x:", logys2.mean(), stderr(logys2))


# do the same for an x that is based on the total number of incoming transmissions per node:

target_list = f.model.transmissions_array[:, 3]
values, counts = np.unique(target_list, return_counts=True)
weights = np.zeros(n_inputs)
weights[values] = counts
shares = weights / weights.sum()
total_budget = 1.0 * n_inputs

x3 = shares * total_budget
ys3 = np.array([f.evaluate(x3, **evaluation_parms) for it in range(n_trials)])
logys3 = np.log(ys3)
print("\nMean and std.err. of", n_trials, "evaluations at a transmissions-based x:", ys3.mean(), stderr(ys3))
print("Mean and std.err. of the log of", n_trials, "evaluations at that x:", logys3.mean(), stderr(logys3))


# do the same for an x that is based on the static network's node degrees:

weights = np.array(list([d for n, d in f.network.degree()]))
shares = weights / weights.sum()

x4 = shares * total_budget
ys4 = np.array([f.evaluate(x4, **evaluation_parms) for it in range(n_trials)])
logys4 = np.log(ys4)
print("\nMean and std.err. of", n_trials, "evaluations at a degree-based x:", ys4.mean(), stderr(ys4))
print("Mean and std.err. of the log of", n_trials, "evaluations at that x:", logys4.mean(), stderr(logys4))


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
