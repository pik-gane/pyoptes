'''
Computes the objective function value of the prior for each strategy and the standard error of the mean for each strategy.
'''
import numpy as np
import glob
import pylab as plt
import os
from pyoptes import bo_compute_average_otf_and_stderr


# TODO can probably be moved into bo_postprocessing
def bbo_inspect_prior(path_plot):

    path_to_numpy_files = glob.glob(os.path.join(path_plot, '**/list_all_prior_tf.npy'), recursive=True)

    print(path_to_numpy_files)

    for path in path_to_numpy_files:
        list_all_prior_tf = np.load(path)
        path2 = os.path.join(os.path.split(path)[0], 'list_all_prior_stderr.npy')

        list_all_prior_stderr = np.load(path2)

        print(list_all_prior_tf.shape, list_all_prior_stderr.shape)

        av_prior_tf, av_prior_stderr = bo_compute_average_otf_and_stderr(list_otf=list_all_prior_tf,
                                                                         list_stderr=list_all_prior_stderr,
                                                                         n_runs=len(list_all_prior_tf))

        print(av_prior_tf.shape, av_prior_stderr.shape)

        path_experiment = os.path.split(os.path.split(path)[0])[0]

        plt.bar(range(len(av_prior_tf)), av_prior_tf, label='prior')
        plt.title(f'Objective function evaluation for {len(av_prior_tf)} strategies,\naverage over {len(list_all_prior_tf)} networks')
        plt.xlabel('Prior')
        plt.ylabel('Number of infected animals')
        # TODO move text in the top right corner of the plot
        # plt.text(25, 1000, f'min: {min_y_prior_mean:2f}\nmax: {max_y_prior_mean:2f}',
        #          bbox=dict(facecolor='red', alpha=0.5))
        plt.savefig(os.path.join(path_experiment, 'objective_function_values_prior.png'))
        plt.clf()