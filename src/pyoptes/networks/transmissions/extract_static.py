from networkx import DiGraph

from .transmissions_data import Transmissions

def extract_static(transmissions=None, transmissions_data=None):
    if transmissions:
        assert transmissions_data is None
        transmissions_data = transmissions.get_data_array()
    
    edges = [(row[2], row[3]) for row in transmissions_data]
    network = DiGraph(edges)
    
    return network