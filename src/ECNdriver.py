'''
Adapted from Matlab code
'''

import numpy as np


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
            # print(a, fval[i+1], fval[i])
            fval[i] = a

        # print(fval[0:nf-1].size - np.count_nonzero(fval[0:nf-1]), nf/2)
        # print(fval[0:nf-1].sum())
        # if j == 0 and fval[0:nf-1].size - np.count_nonzero(fval[0:nf-1]) >= nf/2:
        #     inform = 2
        #     return fnoise, level, inform

        gamma = 0.5*((j+1)/(2*j))*gamma
        print(gamma)
    print('post loop', fval)
    # If noise not detected then h is too large
    inform = 3

    return fnoise, level, inform


if __name__ == '__main__':

    # fval = np.array(range(9))
    fval = np.array([2.0042, 2.0042, 2.0042, 2.0042, 2.0042, 2.0042, 2.0042, 2.0042, 2.0042])

    fnoise, level, inform = ECNoise(9, fval)

    print('fnoise', fnoise)
    print('level', level)
    print('inform', inform)



