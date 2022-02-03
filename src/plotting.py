"""
Simple test to illustrate how the target function could be used.
Here focussing on the 95th PERCENTILE of the no. of infected animals,
and using a simple 2d lattice network of size 10 x 10
"""

import networkx as nx
import numpy as np
from pyparsing import nested_expr
from scipy.stats import gaussian_kde as kde
import pylab as plt
from pyoptes import set_seed
from pyoptes.optimization.budget_allocation import target_function as f

print(f.__doc__)

# set some seed to get reproducible results:
set_seed(1)

print("Preparing the target function for a lattice-based, fixed transmissions network")

net_x = 30
net_y = 30
net_size = net_x*net_y

# generate a 11 by 11 2d lattice with nodes numbered 0 to 120:
lattice = nx.DiGraph(nx.to_numpy_array(nx.lattice.grid_2d_graph(net_x, net_y)))

print(lattice)

# at the beginning, call prepare() once:
f.prepare(
    n_nodes = 900,
    static_network = None,  
    capacity_distribution=np.random.lognormal,  # all nodes have unit capacity
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
#plt.figure()
#nx.draw_kamada_kawai(lattice, node_color=[[0,0,0,xi/xmax] for xi in x])
#plt.show()

# evaluate f once at that input:
y = f.evaluate(
        x, 
        n_simulations=100, 
        statistic=lambda a: np.mean(a**2)  # to focus on the tail of the distribution
        )

print("\nOne evaluation at random x:", y)


evaluation_parms = { 
        'n_simulations': 100, 
        'statistic': lambda a: np.mean(a**2)
        }

n_trials = 1000

def stderr(a):
    return np.std(a, ddof=1) / np.sqrt(np.size(a))


# evaluate f a number of times at the same input:
ys_rnd = np.array([f.evaluate(x, **evaluation_parms) for it in range(n_trials)])
logys_rnd = np.log(ys_rnd)
print("Mean and std.err. of", n_trials, "evaluations at the same random x:", ys_rnd.mean(), stderr(ys_rnd))
print("Mean and std.err. of the log of", n_trials, "evaluations at that x:", logys_rnd.mean(), stderr(logys_rnd))


# do the same for an x that is based on the total capacity of a node:

weights = f.capacities
shares = weights / weights.sum()

x2_cap = shares * total_budget
x2max_cap = x2_cap.max()
#plt.figure()
#nx.draw_kamada_kawai(lattice, node_color=[[0,0,0,xi/x2max_cap] for xi in x2_cap])
#plt.show()

ys2_cap = np.array([f.evaluate(x2_cap, **evaluation_parms) for it in range(n_trials)])
logys2_cap = np.log(ys2_cap)
print("\nMean and std.err. of", n_trials, "evaluations at a capacity-based x:", ys2_cap.mean(), stderr(ys2_cap))
print("Mean and std.err. of the log of", n_trials, "evaluations at that x:", logys2_cap.mean(), stderr(logys2_cap))


# do the same for an x that is based on the total number of incoming transmissions per node:

target_list = f.model.transmissions_array[:, 3]
values, counts = np.unique(target_list, return_counts=True)
weights = np.zeros(n_inputs)
weights[values] = counts
shares = weights / weights.sum()
total_budget = 1.0 * n_inputs

x3_trans = shares * total_budget
x3max_trans = x3_trans.max()
#plt.figure()
#nx.draw_kamada_kawai(lattice, node_color=[[0,0,0,xi/x3max_trans] for xi in x3_trans])
#plt.show()

ys3_trans = np.array([f.evaluate(x3_trans, **evaluation_parms) for it in range(n_trials)])
logys3_trans = np.log(ys3_trans)
print("\nMean and std.err. of", n_trials, "evaluations at a transmissions-based x:", ys3_trans.mean(), stderr(ys3_trans))
print("Mean and std.err. of the log of", n_trials, "evaluations at that x:", logys3_trans.mean(), stderr(logys3_trans))

# do the same for an x that concentrates the budget on a few nodes:

sentinels = list(np.random.choice(np.arange(0,net_size), np.int(np.round(1/5*net_size)), replace=False))

weights = np.zeros(net_size)
weights[sentinels] = 1
shares = weights / weights.sum()

x1 = shares * total_budget
x1max = x1.max()
#plt.figure()
#nx.draw_kamada_kawai(lattice, node_color=[[0,0,0,xi/x1max] for xi in x1])
#plt.show()

ys = np.array([f.evaluate(x1, **evaluation_parms) for it in range(n_trials)])
logys = np.log(ys)
print("\nMean and std.err. of", n_trials, "evaluations at a 1/5 sentinel-based x:", ys.mean(), stderr(ys))
print("Mean and std.err. of the log of", n_trials, "evaluations at that x:", logys.mean(), stderr(logys))

sentinels = list(np.random.choice(np.arange(0,net_size), np.int(np.round(2/5*net_size)), replace=False))

weights = np.zeros(net_size)
weights[sentinels] = 1
shares = weights / weights.sum()

x2 = shares * total_budget
x2max = x2.max()
#plt.figure()
#nx.draw_kamada_kawai(lattice, node_color=[[0,0,0,xi/x2max] for xi in x2])
#plt.show()

ys2 = np.array([f.evaluate(x2, **evaluation_parms) for it in range(n_trials)])
logys2 = np.log(ys2)
print("\nMean and std.err. of", n_trials, "evaluations at a 2/5 sentinel-based x:", ys2.mean(), stderr(ys2))
print("Mean and std.err. of the log of", n_trials, "evaluations at that x:", logys2.mean(), stderr(logys2))

sentinels = list(np.random.choice(np.arange(0,net_size), np.int(np.round(3/5*net_size)), replace=False))

weights = np.zeros(net_size)
weights[sentinels] = 1
shares = weights / weights.sum()

x3 = shares * total_budget
x3max = x3.max()
#plt.figure()
#nx.draw_kamada_kawai(lattice, node_color=[[0,0,0,xi/x3max] for xi in x3])
#plt.show()

ys3 = np.array([f.evaluate(x3, **evaluation_parms) for it in range(n_trials)])
logys3 = np.log(ys3)
print("\nMean and std.err. of", n_trials, "evaluations at a 3/5 sentinel-based x:", ys3.mean(), stderr(ys3))
print("Mean and std.err. of the log of", n_trials, "evaluations at that x:", logys3.mean(), stderr(logys3))

sentinels = list(np.random.choice(np.arange(0,net_size), np.int(np.round(4/5*net_size)), replace=False))

weights = np.zeros(net_size)
weights[sentinels] = 1
shares = weights / weights.sum()

x4 = shares * total_budget
x4max = x4.max()
#plt.figure()
#nx.draw_kamada_kawai(lattice, node_color=[[0,0,0,xi/x4max] for xi in x4])
#plt.show()

ys4 = np.array([f.evaluate(x4, **evaluation_parms) for it in range(n_trials)])
logys4 = np.log(ys4)
print("\nMean and std.err. of", n_trials, "evaluations at a 4/5 sentinel-based x:", ys4.mean(), stderr(ys4))
print("Mean and std.err. of the log of", n_trials, "evaluations at that x:", logys4.mean(), stderr(logys4))

xs = np.linspace(ys3.min(), ys.max())

plt.figure()
plt.plot(xs, kde(ys_rnd)(xs), label="random")
plt.plot(xs, kde(ys2_cap)(xs), label="based on capacities")
plt.plot(xs, kde(ys3_trans)(xs), label="based on transmission")

plt.plot(xs, kde(ys)(xs), label=f"{np.int(np.round(1/5*net_size))} sentinels")
plt.plot(xs, kde(ys2)(xs), alpha=0.5, label=f"{np.int(np.round(2/5*net_size))} sentinels")
plt.plot(xs, kde(ys4)(xs), alpha=0.5, label=f"{np.int(np.round(3/5*net_size))} sentinels")
plt.plot(xs, kde(ys3)(xs), alpha=0.5, label=f"{np.int(np.round(4/5*net_size))} sentinels")
plt.legend()
plt.title("distribution of f(x) for different fixed inputs x")
#plt.show()

xs = np.linspace(logys3.min(), logys.max())
plt.figure()
plt.plot(xs, kde(logys_rnd)(xs), label="random")
plt.plot(xs, kde(logys2_cap)(xs), label="based on capacities")
plt.plot(xs, kde(logys3_trans)(xs), label="based on transmission")

plt.plot(xs, kde(logys)(xs), label=f"{np.int(np.round(1/5*net_size))} sentinels")
plt.plot(xs, kde(logys2)(xs), alpha=0.5, label=f"{np.int(np.round(2/5*net_size))} sentinels")
plt.plot(xs, kde(logys4)(xs), alpha=0.5, label=f"{np.int(np.round(3/5*net_size))} sentinels")
plt.plot(xs, kde(logys3)(xs), alpha=0.5, label=f"{np.int(np.round(4/5*net_size))} sentinels")
plt.legend()
plt.title("distribution of log(f(x)) for different fixed inputs x")
#plt.show()

plt.show()

