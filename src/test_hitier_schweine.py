"""
*** UNDER CONSTRUCTION! ***
Script to exemplify how to use real-world transmission data
and perform SI simulations on it with SIModelOnTransmissions
"""
from time import time
import numpy as np
import networkx as nx
import pylab as plt

from pyoptes.networks.transmissions.hitier_schweine import load_transdataarray
from pyoptes.epidemiological_models.si_model_on_transmissions import SIModelOnTransmissions
from pyoptes.networks import transmissions2static

start_time = time()
#transmissions = load_transdataarray(max_rows=100, verbose=True)
data_array = load_transdataarray(max_rows=100000, verbose=True, return_object=False)
print("...took", time()-start_time, "seconds")
D = transmissions2static(data_array)
G = nx.Graph(D)

# reduce to skeleton by iteratively removing a lowest-degree node:
# TODO: speed up by using numba and numpy arrays
start_time = time()
target_n = 400
while nx.number_of_nodes(G) > target_n:
    nodes, degrees = np.array(list(G.degree())).T
    lowest = degrees.argmin()
    print(nx.number_of_nodes(G), nodes[lowest], degrees[lowest])
    G.remove_node(nodes[lowest])    
print("...took", time()-start_time, "seconds")

nx.write_gexf(G, "/tmp/test.gexf")
plt.figure()
nx.draw_spring(G, alpha=0.3)
#plt.figure()
#nx.draw_kamada_kawai(G, alpha=0.3)
plt.show()

exit()

n_nodes = 10
transmissions_time_covered = 30
n_forward_transmissions_per_day = 3
n_backward_transmissions_per_day = 1
max_t = 360
expected_total_n_tests = n_nodes

# generate transmissions data:
transmissions = get_scale_free_transmissions_data (
    n_nodes=n_nodes, 
    BA_m=2,
    t_max=transmissions_time_covered, 
    n_forward_transmissions_per_day=n_forward_transmissions_per_day,
    n_backward_transmissions_per_day=n_backward_transmissions_per_day,
    verbose=True, 
    )
assert transmissions.max_delay == 0
assert len(transmissions.events) == transmissions_time_covered * (n_backward_transmissions_per_day + n_forward_transmissions_per_day)  
print("\n   ", transmissions)

# distribute the test budget randomly (rather than intelligently):
weights = np.random.rand(n_nodes)
shares = weights / weights.sum()
expected_n_tests = shares * expected_total_n_tests
daily_test_probabilities = expected_n_tests / max_t

# setup the model:
model = SIModelOnTransmissions(
    n_nodes = n_nodes,
    
    use_transmissions_array = True,
    transmissions_array = transmissions.get_data_array(), 
    transmissions_time_covered = transmissions_time_covered, 
    repeat_transmissions = True,
    
    use_tests_array = False,
    daily_test_probabilities = daily_test_probabilities,
    
    p_infection_from_outside = 0.01,
    p_infection_by_transmission = 0.5,
    p_test_positive = 0.9,
    delta_t_testable = 1,
    delta_t_infectious = 2,
    delta_t_symptoms = 60,
    
    max_t = max_t,
    stop_when_detected = True,
    stopping_delay = 1,
    verbose = True,
    )

# run until detection:
model.run()
infected_when_stopped = model.is_infected[model.t,:]
print("Infected when measures are taken (=loss function):", infected_when_stopped.sum(), "at time", model.t)

# continue until t_max:
model.stop_when_detected = False
model.reset()
model.verbose = False
model.run()
infected_eventually = model.is_infected[model.t,:]
print("\nEventually infected:", infected_eventually.sum(), "at time", model.t)
