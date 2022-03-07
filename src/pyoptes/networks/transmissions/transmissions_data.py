from ...util import *
from .. import Node

if False:
    """
    Class attribute declarations, only listed for documentation purposes,
    not actually specified in class definition since numba does not support
    this so far.
    """
    
    t_sent = None
    """(number) time point of when the transmission left the source node"""
    
    t_received = None
    """(number>=t_sent) time point of when the transmission reached the target node, might be the same as t_sent"""
    
    source = None
    """(Node) where transmission comes from"""
    
    target = None
    """(Node) where transmission goes to"""
    
    size = None
    """(number>0) size of the transmission in relevant units, e.g. number of transferred animals"""
     
# Instead, we specify the types for numba jit compilation as follows:
    
spec = [
    ('t_sent', time_t), 
    ('t_received', time_t),
    ('source', node_t),
    ('target', node_t),
    ('size', nb.int64)    
]

@jitclass(spec)
class TransmissionEvent (object):
    """represents a single transmission"""
    
    def __init__(self, t_sent, t_received, source, target, size):
        self.t_sent = t_sent
        self.t_received = t_received
        self.source = source
        self.target = target
        self.size = size

class Transmissions (object):
    """Represents a list of all transmissions of potentially infectious material between nodes
    that happen within a certain time interval"""
    
    time_covered = None
    """(number) length of the (reception) time interval covered by this list of transmissions"""

    events = None
    """(list of TransmissionEvents) in potentially unordered fashion"""

    def __init__(self, time_covered, events):
        self.time_covered = time_covered
        if isinstance(events[0], TransmissionEvent):
            self.events = events
        else:
            self.events = [TransmissionEvent(t_sent, t_received, source, target, size) for (t_sent, t_received, source, target, size) in events]
            
    @property
    def events_by_time_sent(self):
        """(iterator for TransmissionEvents) ordered ascendingly by t_sent"""
        return sorted(self.events, key=lambda tr: tr.t_sent)

    @property
    def events_by_time_received(self):
        """(iterator for TransmissionEvents) ordered ascendingly by t_received"""
        return sorted(self.events, key=lambda tr: tr.t_received)
        
    @property
    def max_delay(self):
        """maximum difference between t_received and t_sent"""
        return max([e.t_received - e.t_sent for e in self.events])
        
    def get_data_array(self, include_size=False):
        """returns the data in array form suitable for SIModelOnTransmissions"""
        if include_size:
            return np.array([[ev.t_sent, ev.t_received, ev.source, ev.target, ev.size]
                             for ev in self.events_by_time_received])
        else:
            return np.array([[ev.t_sent, ev.t_received, ev.source, ev.target]
                             for ev in self.events_by_time_received])
                         
    def __str__(self):
        return "Transmissions:\n" + "\n".join([
            "     sent on day " + str(ev.t_sent) + ", rcvd on day " + str(ev.t_received) + " from " + str(ev.source) + " to " + str(ev.target)
            for ev in self.events_by_time_received
        ]) + "\n"
