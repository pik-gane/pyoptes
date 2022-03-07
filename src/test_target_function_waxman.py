"""
Simple test to illustrate how the target function could be used.
Here focussing on the 2ND MOMENT of the no. of infected animals,
and using a Waxman network
"""

import networkx as nx
import numpy as np
from scipy.stats import gaussian_kde as kde
import pylab as plt
from progressbar import progressbar

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
    delta_t_symptoms=60,
#    parallel=True
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
ms_est, ms_stderr = f.evaluate(x, n_simulations=100)

rms_est = np.sqrt(ms_est)
rms_stderr = ms_stderr / (2 * np.sqrt(ms_est))  # std.err. of the square root of ms_est = std.err. of ms_est * derivative of the square root function

print("\nOne evaluation at random x:", rms_est, rms_stderr)

# for Malte:
    
def est_prob_and_stderr(is_infected):
    ps = np.mean(is_infected, axis=0)
    stderrs = np.sqrt(ps * (1-ps) / is_infected.shape[0])
    return (ps, stderrs)

# use a "non-aggregating" aggregation function plus the above statistic, to get node-specific infection probabilities:
ps, stderrs = f.evaluate(x, n_simulations=100, aggregation=lambda a: a, statistic=est_prob_and_stderr)

print("node infection probabilities:", ps)
print("corresponding std.errs.:     ", stderrs)

evaluation_parms = { 
        'n_simulations': 100
        }

n_trials = 1000

# evaluate f a number of times at the same input:
ys = np.array([np.sqrt(f.evaluate(x, **evaluation_parms)[0]) for it in progressbar(range(n_trials))])
logys = np.log(ys)
print("Mean and std.dev. of", n_trials, "evaluations at the same random x:", ys.mean(), ys.std())

# do the same for an x that is based on the total capacity of a node:

weights = f.capacities
shares = weights / weights.sum()

x2 = shares * total_budget
x2max = x2.max()
plt.figure()
nx.draw(waxman, node_color=[[0,0,0,xi/x2max] for xi in x2], pos=pos)
#plt.show()

ys2 = np.array([np.sqrt(f.evaluate(x, **evaluation_parms)[0]) for it in range(n_trials)])
logys2 = np.log(ys2)
print("Mean and std.dev. of", n_trials, "evaluations at the same capacity-based x:", ys2.mean(), ys2.std())

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

ys3 = np.array([np.sqrt(f.evaluate(x, **evaluation_parms)[0]) for it in range(n_trials)])
logys3 = np.log(ys3)
print("Mean and std.dev. of", n_trials, "evaluations at the same transmissions-based x:", ys3.mean(), ys3.std())

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
