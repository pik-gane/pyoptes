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
from functools import partial
from multiprocessing import cpu_count, Pool


class TargetFunction:
    def __init__(self,
                 use_real_data=False,
                 static_network=None,
                 n_nodes=60000,
                 max_t=365,
                 expected_time_of_first_infection=30,
                 capacity_distribution=np.random.uniform, # any function accepting a 'size=' parameter
                 delta_t_symptoms=60,
                 p_infection_by_transmission=0.5
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
        @param delta_t_symptoms: (optional int, default: 60) After what time the
        infection should be detected automatically even without a test.
        """

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
                self.network = transmissions.BA_network

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
                self.network = static_network

            assert transmissions.max_delay == transmission_delay
            transmissions_array = transmissions.get_data_array()

            # set random capacities:
            total_capacity = 25e6 * n_nodes/60000  # proportional to German pig trade
            weights = capacity_distribution(size=n_nodes)
            shares = weights / weights.sum()
            self.capacities = shares * total_capacity

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
                    p_infection_by_transmission = p_infection_by_transmission,
                    p_test_positive = 0.99,
                    delta_t_testable = 1,
                    delta_t_infectious = 1,
                    delta_t_symptoms = delta_t_symptoms,

                    max_t = max_t,
                    stop_when_detected = True,
                    stopping_delay = 1,
                    verbose = False,
            )

    def get_n_inputs(self):
        """Get the length of the input vector needed for evaluate(),
        which equals the number of nodes in the underlying network."""
        return self.model.n_nodes

    def simulate_infection(self):
        self.model.reset()
        # run until detection:
        self.model.run()
        # return a vector of bools stating which farms are infected at the end:
        return self.model.is_infected[self.model.t, :]

    def n_infected_animals(self, is_infected):
        return np.sum(self.capacities * is_infected)

    def mean_square_and_stderr(self, n_infected_animals):
        values = n_infected_animals**2
        estimate = np.mean(values, axis=0)
        stderr = np.std(values, ddof=1, axis=0) / np.sqrt(values.shape[0])
        return estimate, stderr

    def task(self, unused_simulation_index, aggregation):
        return aggregation(self.simulate_infection())

    def evaluate(self,
                 budget_allocation,
                 n_simulations=1,
                 aggregation=None,
                 statistic=None,
                 parallel=False,
                 num_cpu_cores=2):
        """Run the SIModelOnTransmissions "n_simulations" times, using the given budget
        allocation, and return the number of nodes infected at the time the
        simulation is stopped. Since the simulated process is a stochastic
        process, this returns a "noisy" evaluation of the "true" target function
        (which is the expected value of this number of infected nodes).

        Examples for 'statistic':
        - np.mean
        - np.median
        - np.max
        - lambda a: np.percentile(a, 95)
        - lambda a: np.mean(a**2)

        @param num_cpu_cores: int, specifies the number of cpus for parallelization. Use -1 to use all cpus.
        @param aggregation: any function converting an array of infection bools into an aggregated "damage"
        @param parallel: (bool) Sets whether the simulations runs are computed in parallel. Default is set to True.
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

        budget_allocation = np.array(budget_allocation)
        assert budget_allocation.size == self.model.daily_test_probabilities.size

        if not aggregation:
            aggregation = self.n_infected_animals
        if not statistic:
            statistic = self.mean_square_and_stderr

        self.model.daily_test_probabilities = budget_allocation / 365

        if parallel:
            # check whether the number of cpus are available, if not use all available cpus
            if num_cpu_cores > cpu_count():
                num_cpu_cores = cpu_count()
            # use all cpus available
            elif num_cpu_cores == -1:
                num_cpu_cores = cpu_count()

            with Pool(num_cpu_cores) as pool:
                results = pool.map(partial(self.task, aggregation=aggregation), range(n_simulations))
        else:
            results = [self.task(sim, aggregation) for sim in range(n_simulations)]

        # return the requested statistic:
        return statistic(np.array(results))
        # Note: other indicators are available via target_function.model
