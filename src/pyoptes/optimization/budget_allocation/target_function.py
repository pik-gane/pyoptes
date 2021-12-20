"""
Noisy target function to be minimized, based on the SIModelOnTransmissions,
using transmissions data generated with get_scale_free_transmissions_data.

Input: a budget allocation, i.e., a vector of nonnegative floats specifying
       the expected number of tests each node will take per year.
       
Output: an estimate of the number of infected animals at the time the simulation 
        is stopped due to the detection of an outbreak either because of a test 
        or because of symptoms. This estimate is noisy because it depends on
        the stochastic infection dynamics simulated by the model.
"""

import numpy as np
from ...epidemiological_models.si_model_on_transmissions import SIModelOnTransmissions

global model, capacities, network


def prepare():
    """Prepare the target function before being able to evaluate it for the 
    first time."""

    use_real_data = False #True
    max_t = 365  # simulate at most for one year
    expected_time_of_first_infection = 30
    
    global model, capacities, network
    
    if use_real_data:
        from pyoptes.networks.transmissions.hitier_schweine import load_transdataarray
        transmissions_array = load_transdataarray(verbose=True)
        transmissions_time_covered = transmissions_array[:,[0,1]].max() + 1
        n_nodes = transmissions_array[:,[2,3]].max() + 1
        # TODO: also read the file that states the nodes' estimated capacities
        
    else:
        from pyoptes.networks.transmissions.scale_free import get_scale_free_transmissions_data
        n_nodes = 60000  # as in HI-Tier German pig trade data
        transmissions_time_covered = 180  # typical lifetime of a pig
        n_total_transmissions = 6e6 * transmissions_time_covered / 1440  # as in HI-Tier German pig trade data
        transmission_delay = 1
    
        # generate transmissions data:
        n_transmissions_per_day = int(n_total_transmissions // transmissions_time_covered)
        n_forward_transmissions_per_day = int(0.75 * n_transmissions_per_day)
        n_backward_transmissions_per_day = n_transmissions_per_day - n_forward_transmissions_per_day
        transmissions = get_scale_free_transmissions_data (
            n_nodes=n_nodes, 
            BA_m=20,
            t_max=transmissions_time_covered, 
            n_forward_transmissions_per_day=n_forward_transmissions_per_day,
            n_backward_transmissions_per_day=n_backward_transmissions_per_day,
            delay=transmission_delay,
            verbose=False
            )
        assert transmissions.max_delay == transmission_delay
        assert len(transmissions.events) == transmissions_time_covered * (n_backward_transmissions_per_day + n_forward_transmissions_per_day)  
        transmissions_array = transmissions.get_data_array()
        network = transmissions.BA_network
        
        # set random capacities:
        total_capacity = 25e6  # as in German pig trade 
        weights = np.random.rand(n_nodes)
        shares = weights / weights.sum()
        capacities = shares * total_capacity
         
    # setup the model:
    p_infection_from_outside = 1 / (n_nodes * expected_time_of_first_infection)
    model = SIModelOnTransmissions(
        n_nodes = n_nodes,
        
        use_transmissions_array = True,
        transmissions_array = transmissions_array, 
        transmissions_time_covered = transmissions_time_covered, 
        repeat_transmissions = True,
        
        use_tests_array = False,
        daily_test_probabilities = np.zeros(n_nodes),
        
        p_infection_from_outside = p_infection_from_outside,
        p_infection_by_transmission = 0.9,
        p_test_positive = 0.99,
        delta_t_testable = 1,
        delta_t_infectious = 1,
        delta_t_symptoms = 30,
        
        max_t = max_t,
        stop_when_detected = True,
        stopping_delay = 1,
        verbose = False,
        )

        
def get_n_inputs():
    """Get the length of the input vector needed for evaluate(),
    which equals the number of nodes in the underlying network."""
    global model
    return model.n_nodes


def evaluate(budget_allocation):
    """Run the SIModelOnTransmissions a single time, using the given budget 
    allocation, and return the number of nodes infected at the time the 
    simulation is stopped. Since the simulated process is a stochastic
    process, this returns a "noisy" evaluation of the "true" target function
    (which is the expected value of this number of infected nodes).
    
    @param budget_allocation: (array of floats) expected number of tests per 
    year, indexed by node
    @return: (int) number of nodes infected at the time the simulation is 
    stopped
    """

    global model, capacities

    model.daily_test_probabilities = budget_allocation / 365
    model.reset()
    # run until detection:
    model.run()
    n_infected_when_stopped = (capacities * model.is_infected[model.t,:]).sum()

    return n_infected_when_stopped
    # Note: other indicators are available via target_function.model
