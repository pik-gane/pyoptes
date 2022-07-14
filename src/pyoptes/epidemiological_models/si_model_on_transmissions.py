from ..util import *

if False:
    """
    Class attribute declarations, only listed for documentation purposes,
    not actually specified in class definition since numba does not support
    this so far.
    """
    
    # SYSTEM STATE (changes during a simulation run):
    
    # current time and node state:
        
    t = None
    """(int) Current day"""
    is_infected = None
    """(2d-array of bool) Entry [t, i] says whether on day t node i was already or got newly infected"""
    have_still_data = None
    """(boolean) Whether the transmissions and test data still covers the current time"""

    # relevant parts of history:
        
    t_first_infection = None
    """(int) Time where the first node got infected, if any, else -1"""
    node_first_infected = None
    """(int) Node id of the first infected node, if any, else -1"""
    t_first_detection = None
    """(int) Time of first detection of an infection, either by a positive test or because of symptoms, if any"""
    node_first_detected_as_infected = None
    """(int) Node id of the first node detected as infected, if any, else -1"""
    detection_was_by_test = None
    """(boolean) Whether the detection was due to a test (if not, then because of symptoms)"""
    was_infected_at_first_detection = None
    """(array of bool) Whether each node was infected at the time of the first detection, if any, else empty array"""
    
    # PARAMETERS (fixed during a simulation run):

    n_nodes = None
    """(int) Number of nodes. Node indices are then 0...n_nodes-1"""
        
    # transmissions:
    
    use_transmissions_array = None
    """(boolean) whether to use a list of transmissions (otherwise transmission probabilities)"""
    transmissions_array = None
    """(optional 2d-array of ints with rows (t_sent, t_received, source, target) sorted by t_received (!), where source, target are node ids) 
    Array of transmissions""" # not yet: """Might alternatively be specified via transmission_probabilities"""
    transmissions_time_covered = None
    """(int) time interval covered by the transmissions data"""
    repeat_transmissions = None
    """(optional bool) Whether to forever repeat the whole list of transmissions after the time it covers ended"""
    transmission_network = None
    """(optional 2d-array of ints with rows (source, target, delay), where source, target are node ids
    and delay >= 0"""
    daily_transmission_probabilities = None
    """(optional 2d-array with rows of floats 0...1): Daily probability that a transmission happens from 
    transmission_network[0] to transmission_network[1] that will have a delay of transmission_network[2]. 
    Alternative to self.transmissions"""
    
    # tests:
    
    use_tests_array = None
    """(boolean) whether to use a list of tests (otherwise test probabilities)"""
    tests = None
    """(optional 2d-array with rows (t, node_id) sorted by t)
    Array of times and nodes to test. Might alternatively be specified via daily_test_probabilities"""
    tests_time_covered = None
    """(int) time interval covered by the tests data"""
    repeat_tests = None
    """(bool) Whether to forever repeat the whole list of tests after the time it covers ends"""
    daily_test_probabilities = None
    """(optional array of floats 0...1)
    Daily probabilities that each node gets tested. Alternative to self.tests"""

    # epidemic parameters:
        
    p_infection_from_outside = None
    """(float 0...1) Probability that a node gets infected from outside during one day"""  
    p_infection_by_transmission = None
    """(float 0...1) Probability that the target node of a transmission gets infected when the source node was infected"""
    p_test_positive = None
    """(float 0...1) Probability that a test at an infected node is indeed positive (=sensitivity of the test)"""
    # Note: we assume tests may be false-negative, i.e., p_test_positive<1. 
    # But we assume tests can NOT be false positive. In other words, a positive test result implies the node is infected. 
    delta_t_testable = None
    """(int >= 0) No. of days between getting infected and becoming positively testable""" 
    delta_t_infectious = None
    """(int >= 0) No. of days between getting infected and becoming infectuous""" 
    delta_t_symptoms = None
    """(int >= 0) no. of days after which an infection is automatically detected without test because of obvious symptoms"""

    # options:

    max_t = None
    """(int) Maximal simulation time"""
    stop_when_detected = None
    """(boolean) Whether to stop a run when an infection is detected"""
    stopping_delay = None
    """(optional int) How many days after the first detection to stop the simulation if stop_when_detected=True"""
    verbose = None
    """(boolean) Whether to print log messages to stdout"""
    
    # AUXILIARY DATA:
        
    _transmissions_time_offset = None
    """(int) time offset for transmissions data"""
    _tests_time_offset = None
    """(int) time offset for test data"""
    _next_transmission_index = None
    """(int) index in transmissions to be processed next"""
    _next_test_index = None
    """(int) index in tests to be processed next"""


# Instead, we specify the types for numba jit compilation as follows:
    
spec = [
# SYSTEM STATE (changes during a simulation run): 
# current time and node state:
    ('t', time_t),
    ('is_infected', nb.boolean[:, :]),
    ('have_still_data', nb.boolean),
# relevant parts of history:
    ('t_first_infection', time_t),
    ('node_first_infected', node_t),
    ('t_first_detection', time_t),
    ('node_first_detected_as_infected', node_t),
    ('detection_was_by_test', nb.boolean),
# PARAMETERS (fixed during a simulation run):
    ('n_nodes', node_t),
# transmissions:
    ('use_transmissions_array', nb.boolean),
    ('transmissions_array', nb.int64[:, :]),
    ('transmissions_time_covered', time_t),
    ('_n_edges', nb.int64),
    ('repeat_transmissions', nb.boolean),
    ('transmission_network', nb.int64[:, :]),
    ('daily_transmission_probabilities', prob_t[:]),
# tests:    
    ('use_tests_array', nb.boolean),
    ('tests', nb.int64[:, :]),
    ('tests_time_covered', time_t),
    ('repeat_tests', nb.boolean),
    ('daily_test_probabilities', prob_t[:]),
# epidemic parameters:    
    ('p_infection_from_outside', prob_t),
    ('p_infection_by_transmission', prob_t),
    ('p_test_positive', prob_t),
    ('delta_t_testable', time_t),
    ('delta_t_infectious', time_t),
    ('delta_t_symptoms', time_t),
# simulation options:
    ('max_t', time_t),
    ('stop_when_detected', nb.boolean),
    ('stopping_delay', time_t),
# AUXILIARY DATA:    
    ('_transmissions_time_offset', time_t), 
    ('_tests_time_offset', time_t), 
    ('_next_transmission_index', nb.int64),
    ('_next_test_index', nb.int64),
    ('verbose', nb.boolean),
]


# HERE COMES THE ACTUAL CLASS:

@jitclass(spec) 
class SIModelOnTransmissions (object):
    """Can run simulations of the SI (susceptible-infectious) model on a 
    transmission network."""
    
    # (ATTRIBUTE DECLARATIONS OMITTED, SEE ABOVE)
    
    # CONSTRUCTOR:
        
    def __init__(self, 
                 n_nodes=0,
                 
                 use_transmissions_array=False,
                 transmissions_array=np.array([[-1,-1,-1]]), 
                 transmissions_time_covered=0, 
                 repeat_transmissions=False,
                 transmission_network=np.array([[-1,-1,-1]]),
                 daily_transmission_probabilities=np.array([-1.0]),
                 
                 use_tests_array=False,
                 tests=np.array([[-1,-1]]), 
                 tests_time_covered=0, 
                 repeat_tests=False, 
                 daily_test_probabilities=np.array([-1.0]),
                 
                 p_infection_from_outside=0.0,
                 p_infection_by_transmission=1.0,
                 p_test_positive=1.0,
                 delta_t_testable=0,
                 delta_t_infectious=0,
                 delta_t_symptoms=0,
                 
                 max_t=1,
                 stop_when_detected=True,
                 stopping_delay=0,
                 verbose=False,
                 ):
        """Instantiate a new simulation model and set its parameters"""
        if verbose: print("\n    Setting up SI simulations on transmissions data") 
        # validate arguments:
        assert n_nodes > 0
        self.n_nodes = n_nodes
        
        # are transmissions specified via transmissions or via daily_test_probabilities:
        self.use_transmissions_array = use_transmissions_array
        if use_transmissions_array:
            if verbose: print("     using a list of transmissions")
            self.transmissions_array = transmissions_array
            self.transmissions_time_covered = transmissions_time_covered
            self.repeat_transmissions = repeat_transmissions
        else:
            if verbose: print("     using a directed transmission network and a list of daily transmission probabilities per edge")
            self.transmission_network = transmission_network
            self.daily_transmission_probabilities = daily_transmission_probabilities
            self._n_edges = transmission_network.shape[0]
            
        # are tests specified via tests or via daily_test_probabilities:
        self.use_tests_array = use_tests_array
        if self.use_tests_array:
            if verbose: print("     using a list of tests")
            self.tests = tests
            self.tests_time_covered = tests_time_covered
            self.repeat_tests = repeat_tests
        else: # via daily_test_probabilities
            if verbose: print("     using a list of daily test probabilities per node")
            self.daily_test_probabilities = daily_test_probabilities

        # epidemic parameters:
        assert 0 <= p_infection_from_outside and p_infection_from_outside <= 1
        self.p_infection_from_outside = p_infection_from_outside
        assert 0 <= p_infection_by_transmission and p_infection_by_transmission <= 1
        self.p_infection_by_transmission = p_infection_by_transmission
        assert 0 <= p_test_positive and p_test_positive <= 1
        self.p_test_positive = p_test_positive
        assert 0 <= delta_t_testable
        self.delta_t_testable = delta_t_testable
        assert 0 <= delta_t_infectious
        self.delta_t_infectious = delta_t_infectious
        assert 0 <= delta_t_symptoms
        self.delta_t_symptoms = delta_t_symptoms

        assert 1 <= max_t
        self.max_t = max_t
        self.stop_when_detected = stop_when_detected
        self.stopping_delay = stopping_delay
        self.verbose = verbose
        
        # finally reset to be ready for run():
        self.reset()
        
    # PUBLIC METHODS:
        
    def reset (self):
        """Reset the simulation to time t, no nodes infected, and no past positive test"""
        if self.verbose: print("\n    Resetting SI simulation to day zero") 
        self.t = -1
        self.is_infected = np.zeros((self.max_t + 1, self.n_nodes), nb.boolean)
        self.have_still_data = True
        self.t_first_infection = -1
        self.node_first_infected = -1
        self.t_first_detection = -1
        self.node_first_detected_as_infected = -1
        self._next_transmission_index = -1
        self._next_test_index = -1
        self._transmissions_time_offset = 0
        self._tests_time_offset = 0

    def run (self, n_days=1e6):
        """Run the simulation for n_days 
        (or until an infection is detected if self.stop_when_detected,
        or when running out of transmissions or test data)
        @param n_days: (1...np.inf, default: np.inf) no. of days to run 
        """
        stop_t = min(self.t + n_days, self.max_t)
        if self.verbose: print("\n    Resuming SI simulation from time", self.t+1, "to time", stop_t) 
        
        while self.t < stop_t:
            self.t += 1
            if self.verbose: print("     Day", self.t)
            
            # MORNING of day t:

            # let each node get infected from outside with the set probability:
            rands = np.random.rand(self.n_nodes)
            for node in range(self.n_nodes):
                if rands[node] < self.p_infection_from_outside:
                    # node gets infected from outside
                    if self.verbose: print("      Infection from outside at node", node)
                    self._memorize_infection(node)
                    
            # DURING THE DAY: possible detection:
                
            # check whether an infection is automatically detected because of positive symptoms:
            if self.t_first_infection != -1 and self.t >= self.t_first_infection + self.delta_t_symptoms:
                if self.verbose: print("      Detection because of symptoms", self.delta_t_symptoms, "days after first infection")
                self._memorize_detection(node, was_by_test=False)
                
            # take tests (if no positive test yet, otherwise skip):
            if self.t_first_detection == -1 and self.t >= self.delta_t_testable:
                if self.use_tests_array:
                    if self.repeat_tests:
                        # if necessary, forward offset to cover current time:
                        while self._tests_time_offset + self.tests_time_covered <= self.t:
                            if self.verbose: print("      Reusing tests data for another repetition")
                            self._tests_time_offset += self.tests_time_covered
                            self._next_test_index = -1
                    # is there another test scheduled?
                    if self._tests_time_offset + self.tests[self.tests.shape[0]-1, 0] >= self.t:
                        # yes, so advance test list to current day:
                        next_t_test = -1
                        while next_t_test < self.t:
                            self._next_test_index += 1
                            next_t_test = self._tests_time_offset + self.tests[self._next_test_index, 0]
                        # now process all tests scheduled for this day:
                        while next_t_test == self.t:
                            node = self.tests[self._next_test_index, 1]
                            self._test(node)
                            # advance to next test:
                            self._next_test_index += 1
                            next_t_test = self._tests_time_offset + self.tests[self._next_test_index, 0]
                        # at this point, last extracted test was not performed because it is in the future
                        # hence we will have to extract it again tomorrow:
                        self._next_test_index -= 1
                else:
                    # test each node with a certain probability
                    rands = np.random.rand(self.n_nodes)
                    for node in range(self.n_nodes):
                        if rands[node] < self.daily_test_probabilities[node]:
                            self._test(node)

            if self.stop_when_detected and self.t_first_detection != -1 and self.t >= self.t_first_detection + self.stopping_delay:
                if self.verbose: print("      Stopping on day", self.t, "because of detection on day", self.t_first_detection, "\n")
                return

            # EVENING of day t: transmissions arrive:
                
            if self.use_transmissions_array:
                if self.repeat_transmissions:
                    # if necessary, forward offset to cover current time:
                    while self._transmissions_time_offset + self.transmissions_time_covered <= self.t:
                        if self.verbose: print("      Reusing transmissions data for another repetition")
                        self._transmissions_time_offset += self.transmissions_time_covered
                        self._next_transmission_index = -1
                # is there another transmission scheduled?
                if self._transmissions_time_offset + self.transmissions_array[self.transmissions_array.shape[0]-1, 1] >= self.t:
                    # yes, so advance transmission list to current day:
                    next_t_received = -1
                    while next_t_received < self.t:
                        self._next_transmission_index += 1
                        next_t_received = self._transmissions_time_offset + self.transmissions_array[self._next_transmission_index, 1]
                    # now process all transmissions scheduled for this day:
                    while next_t_received == self.t:
                        t_sent = self._transmissions_time_offset + self.transmissions_array[self._next_transmission_index, 0]
                        source = self.transmissions_array[self._next_transmission_index, 2]
                        target = self.transmissions_array[self._next_transmission_index, 3]
                        self._process_transmission(t_sent, source, target)
                        # advance to next transmission:
                        self._next_transmission_index += 1
                        if self._next_transmission_index < self.transmissions_array.shape[0]:
                            next_t_received = self._transmissions_time_offset + self.transmissions_array[self._next_transmission_index, 1]
                        else:
                            # moved past end of ransmissions list, so next transmission, if any, is not on same day.
                            # hence we can simply increment it since it will be overwritten next time anyway:
                            next_t_received += 1
                    # at this point, last extracted transmission was not performed because it is in the future
                    # hence we will have to extract it again tomorrow:
                    self._next_transmission_index -= 1
                else:
                    # receive a transmission along each edge of the network with a certain probability:
                    rands = np.random.rand(self._n_edges)
                    for edge_index in range(self._n_edges):
                        if rands[edge_index] < self.daily_transmission_probabilities[edge_index]:
                            t_sent = self.t - self.transmission_network[edge_index, 2]
                            source = self.transmission_network[edge_index, 0]
                            target = self.transmission_network[edge_index, 1]
                            self._process_transmission(t_sent, source, target)
            
        if self.verbose:
            print("    Stopping at time", stop_t)
            if self.t == self.max_t: print("     (max_t reached)")
            print()
        return

    # PRIVATE METHODS:
        
    def _test (self, node):
        """simulates a test at the current time a certain node"""
        if self.is_infected[self.t - self.delta_t_testable, node]:
            # node was infected for long enough to be tested positive 
            if np.random.rand() < self.p_test_positive:
                # test is positive
                if self.verbose: print("      Tested node", node, "positive")
                self._memorize_detection(node, was_by_test=True)
            elif self.verbose: print("      Tested node", node, "negative")

    def _process_transmission(self, t_sent, source, target):
            if self.verbose: print("      Transmission from", source, "to", target)
            if (t_sent - self.delta_t_infectious >= 0 
                and self.is_infected[t_sent - self.delta_t_infectious, source]
                and not self.is_infected[self.t, target]):
                # source node was infected for long enough before transmission was sent
                # for the transmission to be potentially infectious,
                # and target node is not infected yet, so test for new infection:
                if np.random.rand() < self.p_infection_by_transmission:
                    # infection was transmitted
                    if self.verbose: print("       causes an infection")
                    self._memorize_infection(target)

    def _memorize_infection (self, node):
        # mark as infected from this day on forever:
        self.is_infected[self.t:, node] = True
        if self.t_first_infection == -1:
            if self.verbose: print("       This is the first infection")
            self.t_first_infection = self.t
            self.node_first_infected = node
            
    def _memorize_detection (self, node, was_by_test):
        if self.t_first_detection == -1:
            if self.verbose: print("       This is the first detection")
            self.t_first_detection = self.t
            self.node_first_detected_as_infected = node
            self.detection_was_by_test = was_by_test
    
