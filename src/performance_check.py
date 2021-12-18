"""
Simple script to assess base performance of SIModelOnTransmissions 
with realistic network size
"""

from time import time
import numpy as np
from pyoptes import set_seed
from pyoptes.networks.transmissions.scale_free import get_scale_free_transmissions_data
from pyoptes.epidemiological_models.si_model_on_transmissions import SIModelOnTransmissions

#set_seed(1)

# PARAMETERS:
    
n_nodes = 60000  # as in HI-Tier German pig trade data
max_t = 365  # simulate at most for one year
expected_total_n_tests = n_nodes * max_t / 365  # i.e. on average one test per node and year
transmissions_time_covered = 180  # typical lifetime of a pig
n_total_transmissions = 6e6 * transmissions_time_covered / 1440  # as in HI-Tier German pig trade data
transmission_delay = 1
expected_time_of_first_infection = 30

# generate transmissions data:
    
start = time()
n_transmissions_per_day = int(n_total_transmissions // transmissions_time_covered)
n_forward_transmissions_per_day = int(0.75 * n_transmissions_per_day)
n_backward_transmissions_per_day = n_transmissions_per_day - n_forward_transmissions_per_day
transmissions = get_scale_free_transmissions_data (
    n_nodes=n_nodes, 
    BA_m=5,
    t_max=transmissions_time_covered, 
    n_forward_transmissions_per_day=n_forward_transmissions_per_day,
    n_backward_transmissions_per_day=n_backward_transmissions_per_day,
    delay=transmission_delay,
    verbose=True
    )
duration = time() - start
print("Generating transmissions data took", duration, "seconds total,", duration/n_total_transmissions, "seconds/transmission")
assert transmissions.max_delay == transmission_delay
assert len(transmissions.events) == transmissions_time_covered * (n_backward_transmissions_per_day + n_forward_transmissions_per_day)  

# distribute the test budget randomly (rather than intelligently):
weights = np.random.rand(n_nodes)
shares = weights / weights.sum()
expected_n_tests = shares * expected_total_n_tests
daily_test_probabilities = expected_n_tests / max_t

# setup the model:
p_infection_from_outside = 1 / (n_nodes * expected_time_of_first_infection)
model = SIModelOnTransmissions(
    n_nodes = n_nodes,
    
    use_transmission_list = True,
    transmissions = transmissions.get_data_array(), 
    transmissions_time_covered = transmissions_time_covered, 
    repeat_transmissions = True,
    
    use_test_list = False,
    daily_test_probabilities = daily_test_probabilities,
    
    p_infection_from_outside = p_infection_from_outside,
    p_infection_by_transmission = 0.9,
    p_test_positive = 0.99,
    delta_t_testable = 1,
    delta_t_infectious = 2,
    delta_t_symptoms = 60,
    
    max_t = max_t,
    stop_when_detected = True,
    stopping_delay = 1,
    verbose = False,
    )

# run until detection:
start = time()
model.run()
print("\nRunning until detection took", time()-start, "seconds")
infected_when_stopped = model.is_infected[model.t,:]
print("Time of first infection:", model.t_first_infection)
print("Was the outbreak detected by a test?", model.detection_was_by_test)
print("Infected when measures are taken (=loss function):", infected_when_stopped.sum(), "at time", model.t)

# continue until t_max:
model.stop_when_detected = False
model.reset()
model.verbose = False
model.run()
print("\nRunning until end took", time()-start, "seconds overall")
infected_eventually = model.is_infected[model.t,:]
print("Eventually infected:", infected_eventually.sum(), "at time", model.t)
