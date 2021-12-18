import random as rd
import numpy as np
import numba as nb

@nb.njit
def _set_numba_seed(seed):
    """helper function"""
    rd.seed(seed)
    np.random.seed(seed)

def set_seed(seed):
    """Set a global seed to be used for the python random package and 
    the numpy.random package, in both normal and numba-jitted functions"""
    rd.seed(seed)
    np.random.seed(seed)
    _set_numba_seed(seed)

time_t = nb.int64
node_t = nb.int64
edge_t = nb.types.Tuple((node_t, node_t))
prob_t = nb.float64
