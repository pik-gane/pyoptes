"""
Simple test to illustrate how the target function could be used.
Here focussing on the 2ND MOMENT of the no. of infected animals,
and using a Waxman network
"""

import networkx as nx
import numpy as np
from scipy.stats import gaussian_kde as kde
import pylab as plt
from pyoptes import set_seed
from pyoptes.optimization.budget_allocation import target_function as f

print(f.__doc__)

# set some seed to get reproducible results:
set_seed(1)

print("Preparing the target function for a Waxman-graph-based, fixed transmissions network")

# generate a Waxman graph:
waxman = nx.waxman_graph(120)
pos = dict(waxman.nodes.data('pos'))
# convert into a directed graph:
static_network = nx.DiGraph(nx.to_numpy_array(waxman))

# at the beginning, call prepare() once:
f.prepare(
    static_network=static_network,  
    capacity_distribution=np.random.lognormal,  # this is more realistic than a uniform distribution
    delta_t_symptoms=60
    )
n_inputs = f.get_n_inputs()
print("n_inputs (=number of network nodes):", n_inputs)

total_budget = 1.0 * n_inputs  # i.e., on average, nodes will do one test per year

# generate a random input:
weights = np.random.rand(n_inputs)
shares = weights / weights.sum()
x = shares * total_budget  

# plot it on the network (the darker a node, the higher the budged):
xmax = x.max()
plt.figure()
nx.draw(waxman, node_color=[[0,0,0,xi/xmax] for xi in x], pos=pos)
#plt.show()

# evaluate f once at that input:
y = f.evaluate(
        x, 
        n_simulations=100, 
        statistic=lambda a: np.percentile(a, 95)  # to focus on the tail of the distribution
        )

print("\nOne evaluation at random x:", y)


evaluation_parms = { 
        'n_simulations': 100, 
        'statistic': np.mean #lambda a: np.percentile(a, 95)
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
x2max = x2.max()
plt.figure()
nx.draw(waxman, node_color=[[0,0,0,xi/x2max] for xi in x2], pos=pos)
#plt.show()

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
x3max = x3.max()
plt.figure()
nx.draw(waxman, node_color=[[0,0,0,xi/x3max] for xi in x3], pos=pos)
#plt.show()

ys3 = np.array([f.evaluate(x3, **evaluation_parms) for it in range(n_trials)])
logys3 = np.log(ys3)
print("\nMean and std.err. of", n_trials, "evaluations at a transmissions-based x:", ys3.mean(), stderr(ys3))
print("Mean and std.err. of the log of", n_trials, "evaluations at that x:", logys3.mean(), stderr(logys3))


xs = np.linspace(ys3.min(), ys.max())
plt.figure()
plt.plot(xs, kde(ys)(xs), label="random x")
plt.plot(xs, kde(ys2)(xs), alpha=0.5, label="capacity-based x")
plt.plot(xs, kde(ys3)(xs), alpha=0.5, label="transmissions-based x")
plt.legend()
plt.title("distribution of f(x) for different fixed inputs x")
#plt.show()

xs = np.linspace(logys3.min(), logys.max())
plt.figure()
plt.plot(xs, kde(logys)(xs), label="random x")
plt.plot(xs, kde(logys2)(xs), alpha=0.5, label="capacity-based x")
plt.plot(xs, kde(logys3)(xs), alpha=0.5, label="transmissions-based x")
plt.legend()
plt.title("distribution of log(f(x)) for different fixed inputs x")
#plt.show()

plt.show()

