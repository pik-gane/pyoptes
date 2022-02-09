'''
Adapted from Matlab code
'''

import numpy as np

from pyoptes import set_seed
from pyoptes.optimization.budget_allocation import target_function as f


def ECNoise(nf, fval):
    """

    @param nf:
    @param fval:
    @return:
    """
    level = np.zeros(nf-1)
    dsgn = np.zeros(nf-1)
    fnoise = 0.0
    gamma = 1.0

    # compute the range of the function
    fmin = fval.min()
    fmax = fval.max()
    if (fmax - fmin) / max(abs(fmax), abs(fmin)) > .1:
        inform = 3
        return fnoise, level, inform
    print('pre loop', fval)
    for j in range(nf-1):
        for i in range(nf-j-1):
            a = fval[i+1] - fval[i]
            fval[i] = a

        if j == 0 and fval[0:nf-1].size - np.count_nonzero(fval[0:nf-1]) >= nf/2:
            inform = 2
            return fnoise, level, inform

        jj = j+1
        gamma = 0.5*((jj)/(2*jj-1))*gamma

        # Compute the estimates for the noise level
        level[j] = np.sqrt(gamma*np.mean(fval[0:nf-j]**2))

        # determine differences in sign
        emin = fval[0:nf-j].min()
        emax = fval[0:nf-j].max()
        if emin*emax < 0.0:
            dsgn[j] = 0

    for k in range(nf-3):
        emin = level[k:k+2].min()
        emax = level[k:k+2].max()
        if emax <= 4*emin and dsgn[k]:
            fnoise = level[k]
            inform = 1
            return fnoise, level, inform

    # If noise not detected then h is too large
    inform = 3
    return fnoise, level, inform


def ECNdriver(n, m, h):

    # define the number of variables
    n = 10
    xb = np.random.rand(n)

    p = np.random.rand(n)
    for i in range(500):
        if np.linalg.norm(p) > 1:
            p = np.random.rand(n)
        else:
            print(p, p.sum(), np.linalg.norm(p))
            break
    p = p/np.linalg.norm(p)

    # Define the number of additional evaluations
    m = 8

    # Define the sampling distance
    h = 1e-14

    fval = np.zeros(m+1)
    mid = np.floor((m+2)/2) # compute half of m, rounded down

    for i in range(0, m+1):

        s = 2*(i-mid)/m
        x = xb + s*h*p
        fval[i] = np.linalg.norm(x)#2.0042

    fnoise, level, inform = ECNoise(9, fval)

    rel_noise = fnoise/mid

    return fnoise, level, inform, rel_noise


if __name__ == '__main__':

    fnoise, level, inform, rel_noise = ECNdriver(n=10, m=8, h=1e-14)
    print('fnoise', fnoise)
    print('level', level)
    print('inform', inform)
    print('rel_noise', rel_noise)
