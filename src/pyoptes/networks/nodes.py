class Node (object):
    """Represents an epidemiological unit, e.g., a person, an animal, or a barn, 
    that may be in a number of epidemiological states, e.g. susceptible, infectious, etc.,
    and that is interpreted as a node in a relevant network of, e.g., acquaintances,
    contacts, transmissions, etc.
    """
    pass
    
    
class Person (Node):
    pass
    
    
class AdministrativeUnit (Node):
    """e.g. a Landkreis"""
    pass
    
    
class Holding (Node):
    """A lifestock holding such as a barn, in which animals are in relatively close contact
    and can mix well"""
    
    type = None
    """(string) Type of holding, e.g. 'breeding', 'fattening', 'trader', 'slaughterhouse'"""
    
    capacity = None
    """(number>0) How many animals can live here at most. Can be used as a weight."""
    
    
