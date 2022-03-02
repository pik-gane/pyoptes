"""
Script to assess numerically the degree of quasi-convexity of the 
target function.
"""

import networkx as nx
import numpy as np
from scipy.stats import gaussian_kde as kde
import pylab as plt
from pyoptes import set_seed
from pyoptes.optimization.budget_allocation import target_function as f

# PARAMETERS:
    
n_farms = 1000 #120
n_simulations = 10000
statistics = lambda n_affected_animals: (
    np.mean(n_affected_animals**2),  # estimated mean of the squared no. affected animals
    np.std(n_affected_animals**2)/np.sqrt(n_simulations)  # std.err. of that estimate
    )
n_testpairs = 100#0


# set some seed to get reproducible results:
set_seed(1)

print("Preparing the target function for a Waxman-graph-based, fixed transmissions network")

# generate a Waxman graph:
waxman = nx.waxman_graph(n_farms)
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
total_budget = 1.0 * n_inputs  # i.e., on average, nodes will do one test per year


convexity_overshoots = []
n_convexity_weak_violations = 0
n_convexity_strong_violations = 0
quasiconvexity_overshoots = []
n_quasiconvexity_weak_violations = 0
n_quasiconvexity_strong_violations = 0
for it in range(n_testpairs):
    # evaluate f at two two random inputs and at their mean:
    weights1 = np.random.lognormal(size=n_inputs)
    weights2 = np.random.lognormal(size=n_inputs)
    shares1 = weights1 / weights1.sum()
    shares2 = weights2 / weights2.sum()
    x1 = shares1 * total_budget  
    x2 = shares2 * total_budget
    xmid = (x1 + x2) / 2
    y1, err1 = f.evaluate(x1, n_simulations=n_simulations, statistic=statistics)
    y2, err2 = f.evaluate(x2, n_simulations=n_simulations, statistic=statistics)
    ymid, errmid = f.evaluate(xmid, n_simulations=n_simulations, statistic=statistics)
    print(it, ":", y1, y2, max(y1, y2), ymid, ymid > max(y1, y2), errmid, ymid - errmid > max(y1 + err1, y2 + err2))
    convexity_overshoots.append(max(0, (ymid - (y1 + y2)/2) / (max(y1, y2) - min(y1, y2))))
    if ymid > (y1 + y2)/2: n_convexity_weak_violations += 1
    if ymid - errmid > (y1 + err1 + y2 + err2)/2: n_convexity_strong_violations += 1
    quasiconvexity_overshoots.append(max(0, (ymid - max(y1, y2)) / (max(y1, y2) - min(y1, y2))))
    if ymid > max(y1, y2): n_quasiconvexity_weak_violations += 1
    if ymid - errmid > max(y1 + err1, y2 + err2): n_quasiconvexity_strong_violations += 1

print("convexity:", np.mean(convexity_overshoots), n_convexity_weak_violations/n_testpairs, n_convexity_strong_violations/n_testpairs)
print("quasi-con:", np.mean(quasiconvexity_overshoots), n_quasiconvexity_weak_violations/n_testpairs, n_quasiconvexity_strong_violations/n_testpairs)

