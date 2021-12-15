
import numpy as np
import numba as nb

if False:
    """
    Class attribute declarations, only listed for documentation purposes,
    not actually specified in class definition since numba does not support
    this so far.
    """
    
    # SYSTEM STATE (changes during a simulation run):
    
    # current time and node state:
        
    t = -1
    """(int) Current day"""
    is_infected = np.array([])
    """(array of bool) Whether each node is currently infected"""

    # relevant parts of history:
        
    t_first_infection = -1
    """(int) Time where the first node got infected, if any, else -1"""
    node_first_infected = -1
    """(int) Node id of the first infected node, if any, else -1"""
    t_first_positive_test = -1
    """(int) Time of first positive test, if any"""
    node_first_tested_positive = -1
    """(int) Node id of the first positive tested node, if any, else -1"""
    was_infected_at_first_positive_test = np.array([])
    """(array of bool) Whether each node was infected at the time of the first positive test, if any, else empty array"""
    
    # PARAMETERS (fixed during a simulation run):
        
    # transmissions:
    
    n_nodes = None
    """(int) Number of nodes. Node indices are then 0...n_nodes-1"""
    transmissions = None
    """(2d-array with rows (t, source, target) sorted by t, where source, target are node ids) 
    Array of transmissions""" # not yet: """Might alternatively be specified via transmission_probabilities"""
    repeat_transmissions = None
    """(bool) Whether to forever repeat the whole list of transmissions after its last entry"""
# not implemented since buggy in numba:
#    transmission_probabilities = None
#    """(optional dict of (source, target): p, where source, target are node ids and 0 <= p <= 1)
#    Daily probability that a transmission happens from source to target. Alternative to self.transmissions"""
    
    # tests:
    
    tests = None
    """(optional 2d-array with rows (t, node_id) sorted by t)
    Array of times and nodes to test. Might alternatively be specified via test_probabilities"""
    repeat_tests = None
    """(bool) Whether to forever repeat the whole dict of tests after the last entry of transmissions (!)"""
    test_probabilities = None
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

    # AUXILIARY DATA:
        
    _transmissions_t_repeat = None
    """(int) day on which transmissions must be repeated, if any"""
    _next_transmission_index = 0
    """(int) index in transmissions to be processed next"""
    _next_test_index = 0
    """(int) index in tests to be processed next"""

# Instead, we specify the types for numba jit compilation as follows:
    
time_t = nb.int64
node_t = nb.int64
edge_t = nb.types.Tuple((node_t, node_t))
prob_t = nb.float64
spec = [
    ('t', time_t),
    ('is_infected', nb.boolean[:]),
    ('t_first_infection', time_t),
    ('node_first_infected', node_t),
    ('t_first_positive_test', time_t),
    ('node_first_tested_positive', node_t),
    ('was_infected_at_first_positive_test', nb.boolean[:]),
    ('n_nodes', node_t),
    ('transmissions', nb.int64[:, :]),
    ('repeat_transmissions', nb.boolean),
    ('use_test_list', nb.boolean),
    ('tests', nb.int64[:, :]),
    ('repeat_tests', nb.boolean),
    ('test_probabilities', prob_t[:]),
    ('p_infection_from_outside', prob_t),
    ('p_infection_by_transmission', prob_t),
    ('p_test_positive', prob_t),
    ('_transmissions_t_repeat', time_t),
    ('_data_time_offset', time_t), 
    ('_next_transmission_index', nb.int64),
    ('_next_test_index', nb.int64),
]


# HERE COMES THE ACTUAL CLASS:
    
@nb.jitclass(spec)
class SIModelOnTransmissions (object):
    """Can run simulations of the SI (susceptible-infectious) model on a 
    transmission network."""
    
    # (ATTRIBUTE DECLARATIONS OMITTED, SEE ABOVE)
    
    # CONSTRUCTOR:
        
    def __init__(self, 
                 n_nodes=0,
                 transmissions=np.array([[-1,-1,-1]]), 
                 repeat_transmissions=False, 
                 tests=np.array([[-1,-1]]), 
                 repeat_tests=False, 
                 test_probabilities=np.array([-1.0]),
                 p_infection_from_outside=0.0,
                 p_infection_by_transmission=1.0,
                 p_test_positive=1.0
                 ):
        """Instantiate a new simulation model and set its parameters"""
        # validate arguments:
        assert n_nodes > 0
        self.n_nodes = n_nodes
        self.transmissions = transmissions
        self.repeat_transmissions = repeat_transmissions
        if repeat_transmissions:
            self._transmissions_t_repeat = transmissions[:, 0].max() + 1
        # are tests specified via tests or via test_probabilities:
        if tests[0,0] != -1:  # via tests
            print("using test list")
            self.use_test_list = True
            self.repeat_tests = repeat_tests
        else: # via test_probabilities
            print("using test probabilities")
            self.use_test_list = False
        self.tests = tests
        self.test_probabilities = test_probabilities
        # epidemic parameters:
        assert 0 <= p_infection_from_outside and p_infection_from_outside <= 1
        self.p_infection_from_outside = p_infection_from_outside
        assert 0 <= p_infection_by_transmission and p_infection_by_transmission <= 1
        self.p_infection_by_transmission = p_infection_by_transmission
        assert 0 <= p_test_positive and p_test_positive <= 1
        self.p_test_positive = p_test_positive
        # finally reset to be ready for run():
        self.reset(0)
        
    # PUBLIC METHODS:
        
    def reset (self, t):
        """Reset the simulation to time t, no nodes infected, and no past positive test"""
        self.t = t
        self.is_infected = np.zeros(self.n_nodes, nb.boolean)
        self.t_first_infection = -1
        self.node_first_infected = -1
        self.t_first_positive_test = -1
        self.node_first_tested_positive = -1
        self.was_infected_at_first_positive_test = np.zeros(self.n_nodes, nb.boolean)
        self._next_transmission_index = 0
        self._next_test_index = 0
        self._data_time_offset = 0

    def run (self, n_days=time_t.maxval, stop_when_positive=True):
        """Run the simulation for n_days
        @param n_days: (1...np.inf, default: np.inf) no. of days to run 
        @param stop_when_positive: (bool, default: True) whether to stop simulation as soon as a test was positive
        """
        stop_t = self.t + n_days
        while self.t < stop_t:
            print("t", self.t)
            
            # morning of day t: take tests (if no positive test yet, otherwise skip):
                
            if self.t_first_positive_test == -1:
                if self.use_test_list:
                    test_offset = self._data_time_offset if self.repeat_tests else 0
                    # is there another test scheduled?
                    if test_offset + self.tests[self.tests.shape[0]-1, 0] >= self.t:
                        # yes, so advance test list to current day:
                        test_t = -1
                        while test_t < self.t:
                            test_t = test_offset + self.tests[self._next_test_index, 0]
                            node = self.tests[self._next_test_index, 1]
                            self._next_test_index += 1
                        # now process all tests on this day:
                        while test_t == self.t:
                            print("testing", node, "infected", self.is_infected[node])
                            if self.is_infected[node]:
                                if np.random.rand() < self.p_test_positive:
                                    print(" positive")
                                    # test is positive
                                    self._memorize_first_positive_test(node)
                                    if stop_when_positive:
                                        print("stopping")
                                        return
                            # advance to next test:
                            test_t = test_offset + self.tests[self._next_test_index, 0]
                            node = self.tests[self._next_test_index, 1]
                            self._next_test_index += 1
                        # at this point, last extracted test was not performed because it is in the future
                        # hence we will have to extract it again tomorrow:
                        self._next_test_index -= 1
                else:
                    # test each node with a certain probability
                    rands = np.random.rand(self.n_nodes)
                    for node in range(self.n_nodes):
                        if rands[node] < self.test_probabilities[node]:
                            print("testing", node, "infected", self.is_infected[node])
                            if self.is_infected[node]:
                                if np.random.rand() < self.p_test_positive:
                                    # test is positive
                                    print(" positive")
                                    self._memorize_first_positive_test(node)
                                    if stop_when_positive:
                                        print("stopping")
                                        return
                            
            # afternoon of day t: transmissions arrive:
                
            if self.repeat_transmissions:
                while self._data_time_offset + self.transmissions[self.transmissions.shape[0]-1, 0] < self.t:
                    print("reusing transmissions data for another repetition")
                    self._data_time_offset += self._transmissions_t_repeat
            # is there another transmission scheduled?
            if self._data_time_offset + self.transmissions[self.transmissions.shape[0]-1, 0] >= self.t:
                # yes, so advance transmission list to current day:
                trans_t = -1
                while trans_t < self.t:
                    trans_t = self._data_time_offset + self.transmissions[self._next_transmission_index, 0]
                    source = self.transmissions[self._next_transmission_index, 1]
                    target = self.transmissions[self._next_transmission_index, 2]
                    self._next_transmission_index += 1
                # now process all transmissions on this day:
                while trans_t == self.t:
                    print("transmitting from", source, "to", target)
                    if self.is_infected[source]:
                        if np.random.rand() < self.p_infection_by_transmission:
                            # infection was transmitted
                            print(" infection transmitted")
                            self.is_infected[target] = True
                            self._memorize_first_infection(node)
                    # advance to next test:
                    trans_t = self._data_time_offset + self.transmissions[self._next_transmission_index, 0]
                    source = self.transmissions[self._next_transmission_index, 1]
                    target = self.transmissions[self._next_transmission_index, 2]
                    self._next_transmission_index += 1
                # at this point, last extracted transmission was not performed because it is in the future
                # hence we will have to extract it again tomorrow:
                self._next_transmission_index -= 1
            
            self.t += 1
        return

    # PRIVATE METHODS:
        
    def _memorize_first_infection (self, node):
        if self.t_first_infection is None:
            self.t_first_infection = self.t
            self.node_first_infected = node
            
    def _memorize_first_positive_test (self, node):
        if self.t_first_positive_test is None:
            self.t_first_positive_test = self.t
            self.node_first_tested_positive = node
            # make a copy of is_infected:
            self.was_infected_at_first_positive_test = self.is_infected.copy()
            
            
            
if __name__ == '__main__':
    print("Testing...")
    m = SIModelOnTransmissions(
        n_nodes=1,
        transmissions=np.array([[0, 0, 0]]), 
        repeat_transmissions=True, 
        tests=np.array([[-1, -1]]), #np.array([[0, 0]]), 
        repeat_tests=True, 
        test_probabilities=np.array([0.1]), #np.array([-1.0]),
        p_infection_from_outside=0.01,
        p_infection_by_transmission=0.5,
        p_test_positive=0.95    
    )
    m.run(100)
    
    