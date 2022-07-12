'''
Compute the target function value

'''

from pyoptes import create_graph, compute_average_otf_and_stderr

import glob
import json
import os
import argparse
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import euclidean
import pylab as plt

if __name__ == '__main__':

    path_to_optimizer = 'pyoptes/optimization/budget_allocation/blackbox_learning/plots/**/evaluation_output.txt'

    #
    optimizer_output = glob.glob(path_to_optimizer)
    print(np.shape(optimizer_output))
    for evaluation_output in optimizer_output:

        with open(evaluation_output, 'r') as f:
            lines = f.readlines()
        print(lines)

        break
