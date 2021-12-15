from .. import Node


class ContactEvent (object):
    """Represents a single contact or meeting between to nodes, happening for some interval of time,
    e.g. two people being closer than 5m for some time."""
    
    time_start = None
    """(number) time point of when the contact started"""
    
    time_end = None
    """(number>=time_start) time point of when the contact ended"""
    
    nodes = None
    """(Node, Node) tuple of the two meeting nodes"""
    
    intensity = None
    """(number>0) intensity of the contact in relevant units, e.g. inverse average spatial distance"""
     

class ContactsList (object):
    """Represents a list of all contacts between nodes
    that happen within a certain time interval"""
    
    time_covered_begin = None
    """(number) begin of the time interval covered by this list of contacts"""

    time_covered_end = None
    """(number) end of the time interval covered by this list of contacts"""
    
    events = None
    """(list of ContactEvents) in unordered fashion"""

    _events_by_time_start = None
    @property
    def events_by_time_start(self):
        """(generator for ContactEvents) ordered ascendingly by time_start"""
        pass

    _events_by_time_end = None
    @property
    def events_by_time_end(self):
        """(generator for ContactEvents) ordered ascendingly by time_end"""
        pass
        
    