from networkx import barabasi_albert_graph
from numpy.random import choice

def get_scale_free_transmissions_data (n_nodes=None, BA_m=None, t_max=None, n_transmissions_per_day=None, BA_seed=None):
    assert isinstance(n_nodes, int)
    assert isinstance(BA_m, int)
    assert isinstance(t_max, int)
    assert t_max > 0
    assert isinstance(n_transmissions_per_day, int)
    
    G = barabasi_albert_graph(n_nodes, BA_m, seed=BA_seed)
    edges = list(G.edges)
    n_edges = len(edges)
    
    transmissions = []
    for t in range(t_max):
        for i in range(n_transmissions_per_day):
            i, j = edges[choice(range(n_edges))]  # chooses a random edge
            source, target = max(i, j), min(i, j)
            transmissions.append([t, source, target, 1])
            
    return transmissions

if __name__=='__main__':
    print(get_scale_free_transmissions_data (n_nodes=100, BA_m=2, t_max=100, n_transmissions_per_day=4))
    
