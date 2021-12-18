from networkx import barabasi_albert_graph
import random as rd
#from numpy.random import choice
from ...util import *
from .transmissions_data import TransmissionEvent, Transmissions

@nb.njit
def _generate_events (
        edges=np.zeros((1,2)),
        t_max=0,
        n_forward_transmissions_per_day=0,
        n_backward_transmissions_per_day=0,
        delay=0,
        verbose=True,   # whether to print information to stdout
        ):
    n_edges = edges.shape[0]
    events = []
    for t in range(t_max):
        if verbose: 
            print("       Day", t)
        for i in range(n_forward_transmissions_per_day):
            pair = edges[rd.randint(0, n_edges-1),:]  # chooses a random edge
            source, target = max(pair), min(pair)  # direction "forward"
            events.append(TransmissionEvent(t - delay, t, source, target, 1))
        for i in range(n_backward_transmissions_per_day):
            pair = edges[rd.randint(0, n_edges-1),:]  # chooses a random edge
            source, target = min(pair), max(pair)  # direction "backward"
            events.append(TransmissionEvent(t - delay, t, source, target, 1))
    return events

def get_scale_free_transmissions_data (
        n_nodes=None, 
        BA_m=None,      # m value for NetworkX.barabasi_albert_graph
        BA_seed=None,   # seed value for NetworkX.barabasi_albert_graph
        t_max=None,
        n_forward_transmissions_per_day=None,
        n_backward_transmissions_per_day=0,
        delay=0,
        verbose=True,   # whether to print information to stdout
        ):
    assert isinstance(n_nodes, int)
    assert isinstance(BA_m, int)
    assert isinstance(t_max, int)
    assert t_max > 0
    assert isinstance(n_forward_transmissions_per_day, int)
    assert isinstance(n_backward_transmissions_per_day, int)
    assert isinstance(delay, int)
    assert delay >= 0
    
    if verbose: 
        print("    Generating scale-free transmissions data")
        print("     n_nodes:", n_nodes)
        print("     Generating a directed and static underlying Barabasi-Albert network")
        print("      outgoing links per node:", BA_m)
        if BA_seed is not None: print("      seed:", BA_seed)
    G = barabasi_albert_graph(n_nodes, BA_m, seed=BA_seed)
    edges = np.array(list(G.edges))
    n_edges = edges.shape[0]
    if verbose:
        print("      ...done. Edges:", n_edges)
        print("     Generating transmission events")
        print("      forward transmissions per day:", n_forward_transmissions_per_day)
        print("      backward transmissions per day:", n_backward_transmissions_per_day)
        print("      delay:", delay)

    events = _generate_events(
        edges=edges,
        t_max=t_max,
        n_forward_transmissions_per_day=n_forward_transmissions_per_day,
        n_backward_transmissions_per_day=n_backward_transmissions_per_day,
        delay=delay,
        verbose=verbose,   # whether to print information to stdout
        )    
    if verbose: 
        print("     Total no. of transmissions:", len(events))
        print("     ...done")

    return Transmissions(t_max, events)
