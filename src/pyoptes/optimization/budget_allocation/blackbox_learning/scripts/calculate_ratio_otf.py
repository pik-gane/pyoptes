'''
@deprecated Fixes errors in the calculation of the ratio of otf to baseline.
'''

import numpy
import os
import glob

if __name__ == '__main__':

    path_to_experiments = glob.glob(os.path.join('../../../../../../data/blackbox_learning/results/',
                                                 '**/evaluation_output.txt'))

    path_to_experiments_old = glob.glob(os.path.join('../../../../../../data/blackbox_learning/results/old/',
                                                     '**/evaluation_output.txt'))
    path_to_experiments.extend(path_to_experiments_old)
    print(f'Number of experiments: {len(path_to_experiments)}')
    for experiment in path_to_experiments:
        # print(experiment)
        with open(experiment, 'r') as f:
            lines = f.readlines()
            # get the line containing the (faulty) otf ratio
            line_ratio_otf = lines[1]
            # split the float value from the line
            recorded_ratio_otf = float(line_ratio_otf.split(':')[-1])

        # create path to raw data from experiment
        path_to_raw_data = os.path.join(os.path.dirname(experiment), 'raw_data')
        # load list_baseline_otf and list_best_otf with numpy
        list_baseline_otf = numpy.load(os.path.join(path_to_raw_data, 'list_baseline_otf.npy'))
        list_best_otf = numpy.load(os.path.join(path_to_raw_data, 'list_best_otf.npy'))

        ratio_otf = []
        for o, b in zip(list_best_otf, list_baseline_otf):
            r = 100 - 100 * o / b
            ratio_otf.append(r)

        mean_ratio_otf = numpy.mean(ratio_otf)

        if recorded_ratio_otf != mean_ratio_otf:

            print('--------------------------------------------')
            print('new ratio', mean_ratio_otf)
            print('old ratio', recorded_ratio_otf)
            print(experiment, '\n')

            # replace the line containing the (faulty) otf ratio with the correct one
            l = lines[1].split(':')[0] + f': {mean_ratio_otf}\n'
            with open(experiment, 'w') as f:
                lines[1] = l
                f.writelines(lines)

        # break
