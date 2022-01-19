import networkx as nx
import random as rd
from ...util import *
from .transmissions_data import TransmissionEvent, Transmissions

@nb.njit
def _generate_events (
        edges=np.zeros((1,2)),
        t_max=0,
        n_transmissions_per_day=0,
        delay=0,
        verbose=True,   # whether to print information to stdout
        ):
    n_edges = edges.shape[0]
    events = []
    for t in range(t_max):
        if verbose: 
            print("       Day", t)
        for i in range(n_transmissions_per_day):
            pair = edges[rd.randint(0, n_edges-1),:]  # chooses a random edge
            source, target = pair[0], pair[1]
            events.append(TransmissionEvent(t - delay, t, source, target, 1))
    return events

def get_static_network_based_transmissions_data (
        network=None,  # a NetworkX Graph or DiGraph
        t_max=None,
        n_transmissions_per_day=None,
        delay=0,
        verbose=True,   # whether to print information to stdout
        ):
    assert isinstance(network, (nx.Graph, nx.DiGraph))
    assert t_max > 0
    assert isinstance(n_transmissions_per_day, int)
    assert isinstance(delay, int)
    assert delay >= 0

    G = nx.DiGraph(network)  # make sure that edges are both ways!    
    n_nodes = G.number_of_nodes()
    
    if verbose: 
        print("    Generating static network-based transmissions data")
        print("     n_nodes:", n_nodes)
    edges = np.array(list(G.edges))
    n_edges = edges.shape[0]
    if verbose:
        print("     n_edges:", n_edges)
        print("     Generating transmission events")
        print("      transmissions per day:", n_transmissions_per_day)
        print("      delay:", delay)

    events = _generate_events(
        edges=edges,
        t_max=t_max,
        n_transmissions_per_day=n_transmissions_per_day,
        delay=delay,
        verbose=verbose, 
        )    
    if verbose: 
        print("     Total no. of transmissions:", len(events))
        print("     ...done")

    tr = Transmissions(t_max, events)
    setattr(tr, 'static_network', G)
    return tr
