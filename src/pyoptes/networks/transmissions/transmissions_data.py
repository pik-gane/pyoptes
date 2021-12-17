import numpy as np
from .. import Node


class TransmissionEvent (object):
    """represents a single transmission"""
    
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
    """(number) length of the time interval covered by this list of transmissions"""

    events = None
    """(list of TransmissionEvents) in potentially unordered fashion"""

    def __init__(self, time_covered, events):
        self.time_covered = time_covered
        self.events = events

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
        
    def get_data_array(self):
        """returns the data in array form suitable for SIModelOnTransmissions"""
        return np.array([[ev.t_sent, ev.t_received, ev.source, ev.target]
                         for ev in self.events_by_time_received])
                         
    def __str__(self):
        return "Transmissions:\n" + "\n".join([
            "    sent " + str(ev.t_sent) + " rcvd " + str(ev.t_received) + " from " + str(ev.source) + " to " + str(ev.target)
            for ev in self.events_by_time_received
        ]) + "\n"
