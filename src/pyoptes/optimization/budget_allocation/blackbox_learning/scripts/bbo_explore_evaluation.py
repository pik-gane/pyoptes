'''
Compare optimizer outputs of the saved in the evaluation_output txt-file with the output in the plot

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


def bbo_explore_evaluation():

    # TODO path is wrong
    path_to_optimizer = 'pyoptes/optimization/budget_allocation/blackbox_learning/plots/**/evaluation_output.txt'

    #
    optimizer_output = glob.glob(path_to_optimizer)
    print(np.shape(optimizer_output))
    for evaluation_output in optimizer_output:

        with open(evaluation_output, 'r') as f:
            lines = f.readlines()
        for l in lines:
            print(l)

        ratio = float(lines[1].split(':')[1])
        baseline_otf, optimizer_stderr = lines[2].split(':')[1].split(', ')
        optimizer_otf, optimizer_stderr = lines[3].split(':')[1].split(', ')
        print(ratio)
        print(baseline)
        print(optimizer_otf, optimizer_stderr)
        break
