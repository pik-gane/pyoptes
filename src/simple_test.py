import numpy as np
from pyoptes.networks.transmissions.scale_free import get_scale_free_transmissions_data
from pyoptes.epidemiological_models.si_model_on_transmissions import SIModelOnTransmissions

n_nodes = 10
transmissions_time_covered = 10
n_transmissions_per_day = 2
max_t = 180
expected_total_n_tests = 100 

# generate transmissions data:
transmissions = get_scale_free_transmissions_data (
    n_nodes=n_nodes, 
    t_max=transmissions_time_covered, 
    n_transmissions_per_day=n_transmissions_per_day,
    BA_m=2, 
    BA_seed=1,
    )
assert transmissions.max_delay == 0
assert len(transmissions.events) == transmissions_time_covered * n_transmissions_per_day   
print(transmissions)

# distribute the test budget randomly:
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
    delta_t_testable = 0,
    delta_t_infectious = 2,
    delta_t_symptoms = 30,
    
    max_t = max_t,
    stop_when_detected = True,
    stopping_delay = 1,
    verbose = True,
    )
model.run(100)
model.reset()
model.verbose = False
model.run(100)

infected_at_end = model.is_infected[model.t,:]
print("Currently infected:", infected_at_end.sum())
