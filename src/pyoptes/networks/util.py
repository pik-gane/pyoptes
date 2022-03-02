import networkx as nx
from .transmissions.transmissions_data import Transmissions

def transmissions2static(transmissions, store=False):
    """Aggregate transmissions data into a weighted static network
    and optionally store it back in the Transmissions object"""
    
    data_array = transmissions.get_data_array(include_size=True) if isinstance(transmissions, Transmissions) else transmissions
    
    weights = {}
    
    print(weights)
    
    for (_, _, i, j, w) in data_array:
        if (i,j) not in weights: 
            weights[(i,j)] = w
        else:
            weights[(i,j)] += w
    
    edge_list = list(weights.keys())
        
    G = nx.DiGraph(edge_list)
    nx.set_edge_attributes(G, weights, "weight")
    
    if store and isinstance(transmissions, Transmissions):
        transmissions.static_network = G
        
    return G
    