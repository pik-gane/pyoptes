from numpy import loadtxt
filename = "/home/heitzig/data/optes/hitier_schweine.csv"

def load_transdataarray(verbose=False):
    if verbose: print("    Loading transmissions data from", filename)
    indata = loadtxt(filename, delimiter=",", skiprows=1, dtype=int)
    return indata[:,[2,2,0,1]]
    if verbose: print("    ...done")
