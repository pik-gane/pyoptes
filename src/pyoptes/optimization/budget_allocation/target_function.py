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

global model, capacities, network, transmissions_array, transmissions_time_covered


def prepare(use_real_data=False, 
            static_network=None,
            n_nodes=60000,
            max_t=365, 
            expected_time_of_first_infection=30, 
            capacity_distribution=np.random.uniform, # any function accepting a 'size=' parameter
            delta_t_symptoms=30
            ):
    """Prepare the target function before being able to evaluate it for the 
    first time.
    
    @param use_real_data: (boolean) Whether to use the real HI-Tier dataset
    (default: false)
    @param static_network: (optional networkx Graph or DiGraph) Which network
    to base transmissions on (if not use_real_data), or None if a scale-free
    network should be used (default: None)
    @param n_nodes: (optional int) Number of nodes for the scale-free network
    (if not use_real_data and static_network is None) (default: 60000)
    @param max_t: (optional int) Maximal simulation time in days (default: 365)
    @param expected_time_of_first_infection: (optional int, default: 30)
    @param delta_t_symptoms: (optional int, default: 30) After what time the 
    infection should be detected automatically even without a test.
    """

    global model, capacities, network, transmissions_array, transmissions_time_covered
    
    if use_real_data:
        from pyoptes.networks.transmissions.hitier_schweine import load_transdataarray
        transmissions_array = load_transdataarray(verbose=True)
        transmissions_time_covered = transmissions_array[:,[0,1]].max() + 1
        n_nodes = transmissions_array[:,[2,3]].max() + 1
        # TODO: also read the file that states the nodes' estimated capacities
        
    else:
        if static_network is not None:
            n_nodes = static_network.number_of_nodes()
            
        transmissions_time_covered = 180  # typical lifetime of a pig
        n_total_transmissions = 6e6 * n_nodes/60000 * transmissions_time_covered/1440  # proportional to HI-Tier German pig trade data
        transmission_delay = 1
    
        # generate transmissions data:
        n_transmissions_per_day = int(n_total_transmissions // transmissions_time_covered)

        if static_network is None:
            # use a scale-free network:
            from pyoptes.networks.transmissions.scale_free import get_scale_free_transmissions_data
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
            assert len(transmissions.events) == transmissions_time_covered * (n_backward_transmissions_per_day + n_forward_transmissions_per_day)  
            network = transmissions.BA_network
            
        else:
            from pyoptes.networks.transmissions.static_network_based import get_static_network_based_transmissions_data
            transmissions = get_static_network_based_transmissions_data (
                network=static_network,
                t_max=transmissions_time_covered, 
                n_transmissions_per_day=n_transmissions_per_day,
                delay=transmission_delay,
                verbose=True #False
                )
            assert len(transmissions.events) == transmissions_time_covered * n_transmissions_per_day  
            network = static_network
            
        assert transmissions.max_delay == transmission_delay
        transmissions_array = transmissions.get_data_array()
        
        # set random capacities:
        total_capacity = 25e6 * n_nodes/60000  # proportional to German pig trade 
        weights = capacity_distribution(size=n_nodes)
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
        delta_t_symptoms = delta_t_symptoms,
        
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


def evaluate(budget_allocation, 
             n_simulations=1, 
             statistic=np.mean  # any function converting an array into a number
             ):
    """Run the SIModelOnTransmissions a single time, using the given budget 
    allocation, and return the number of nodes infected at the time the 
    simulation is stopped. Since the simulated process is a stochastic
    process, this returns a "noisy" evaluation of the "true" target function
    (which is the expected value of this number of infected nodes).
    
    Examples for 'statistic':
    - np.mean
    - np.median
    - np.max
    - lambda a: np.percentile(a, 95)
    
    @param budget_allocation: (array of floats) expected number of tests per 
    year, indexed by node
    @param n_simulations: (optional int) number of epidemic simulation runs the 
    evaluation should be based on (default: 1)
    @param statistic: (optional function array-->float) function aggregating 
    the results from the individual epidemic simulation runs into a single
    number to be used as the evaluation (default: np.mean) 
    @return: (float) typical (in the sense of the requested statistic) number
    of animals infected at the time the simulation is stopped
    """

    global model, capacities

    budget_allocation = np.array(budget_allocation)
    assert budget_allocation.size == model.daily_test_probabilities.size
    
    model.daily_test_probabilities = budget_allocation / 365
    
    n_infected_when_stopped = np.zeros(n_simulations)
    for sim in range(n_simulations):
        model.reset()
        # run until detection:
        model.run()
        # store simulation result:
        n_infected_when_stopped[sim] = (capacities * model.is_infected[model.t,:]).sum()

    # return the requested statistic:
    return statistic(n_infected_when_stopped)
    # Note: other indicators are available via target_function.model
