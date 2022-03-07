from numpy import loadtxt
from .transmissions_data import Transmissions

filename = "/home/heitzig/data/optes/hitier_schweine.csv"

def load_transdataarray(max_rows=None, verbose=False, return_object=True):
    if verbose: print("    Loading transmissions data from", filename)
    indata = loadtxt(filename, delimiter=",", skiprows=1, dtype=int, max_rows=max_rows)
    events = indata[:,[2,2,0,1,3]]
    return Transmissions(indata[:,2].max() + 1, events) if return_object else events
    if verbose: print("    ...done")
