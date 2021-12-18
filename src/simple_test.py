"""
Simple script to exemplify how to 
generate scale-free transmission data with get_scale_free_transmissions_data
and perform SI simulations on it with SIModelOnTransmissions
"""

import numpy as np
from pyoptes.networks.transmissions.scale_free import get_scale_free_transmissions_data
from pyoptes.epidemiological_models.si_model_on_transmissions import SIModelOnTransmissions

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
    
    use_transmission_list = True,
    transmissions = transmissions.get_data_array(), 
    transmissions_time_covered = transmissions_time_covered, 
    repeat_transmissions = True,
    
    use_test_list = False,
    daily_test_probabilities = daily_test_probabilities,
    
    p_infection_from_outside = 0.01,
    p_infection_by_transmission = 0.5,
    p_test_positive = 0.9,
    delta_t_testable = 1,
    delta_t_infectious = 2,
    delta_t_symptoms = 30,
    
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
