from .. import Node


class TransmissionEvent (object):
    """represents a single transmission"""
    
    time_sent = None
    """(number) time point of when the transmission left the source node"""
    
    time_received = None
    """(number>=time_sent) time point of when the transmission reached the target node, might be the same as time_sent"""
    
    source_node = None
    """(Node) where transmission comes from"""
    
    target_node = None
    """(Node) where transmission goes to"""
    
    size = None
    """(number>0) size of the transmission in relevant units, e.g. number of transferred animals"""
     

class TransmissionsList (object):
    """Represents a list of all transmissions of potentially infectious material between nodes
    that happen within a certain time interval"""
    
    time_covered_begin = None
    """(number) begin of the time interval covered by this list of transmissions"""

    time_covered_end = None
    """(number) end of the time interval covered by this list of transmissions"""
    
    events = None
    """(list of TransmissionEvents) in unordered fashion"""

    _events_by_time_sent = None
    @property
    def events_by_time_sent(self):
        """(generator for TransmissionEvents) ordered ascendingly by time_sent"""
        pass

    _events_by_time_received = None
    @property
    def events_by_time_received(self):
        """(generator for TransmissionEvents) ordered ascendingly by time_received"""
        pass
        
    