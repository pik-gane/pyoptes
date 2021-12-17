from networkx import barabasi_albert_graph
from numpy.random import choice
from .transmissions_data import TransmissionEvent, Transmissions

def get_scale_free_transmissions_data (
        n_nodes=None, 
        t_max=None,
        n_transmissions_per_day=None,
        BA_m=None,      # m value for NetworkX.barabasi_albert_graph
        BA_seed=None,   # seed value for NetworkX.barabasi_albert_graph
        ):
    assert isinstance(n_nodes, int)
    assert isinstance(BA_m, int)
    assert isinstance(t_max, int)
    assert t_max > 0
    assert isinstance(n_transmissions_per_day, int)
    
    G = barabasi_albert_graph(n_nodes, BA_m, seed=BA_seed)
    edges = list(G.edges)
    n_edges = len(edges)
    
    events = []
    for t in range(t_max):
        for i in range(n_transmissions_per_day):
            i, j = edges[choice(range(n_edges))]  # chooses a random edge
            source, target = max(i, j), min(i, j)
            events.append(TransmissionEvent(t, t, source, target, 1))
            
    return Transmissions(t_max, events)
    
