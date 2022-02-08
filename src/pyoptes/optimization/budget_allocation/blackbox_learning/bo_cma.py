import cma
import numpy as np


def bo_cma(objective_function, initial_population,
           max_iterations,
           n_simulations,
           indices,
           true_size_x,
           eval_function,
           bounds,
           sigma=0.2):
    """

    @param objective_function: function object
    @param eval_function: function object,
    @param bounds: list,
    @param n_simulations: int,
    @param indices: list,
    @param true_size_x: int,
    @param initial_population: numpy array,
    @param sigma: float,
    @param max_iterations: int,
    @return:
    """
    ea = cma.fmin(objective_function, initial_population, sigma0=sigma,
                  options={'maxiter': max_iterations, 'verbose': -8, 'bounds': bounds},
                  args=(n_simulations, indices, true_size_x, eval_function))

    solutions = ea[-2].pop_sorted

    print('solutions', np.shape(solutions))

    # cma.plot('test')
    # input()
    # logger = ea[-1].load()
    # logger.plot_all()
    # print(solutions)
    # cma.s.figsave('f')
    # cma.plot('outcmaes')
    # print('\nEvaluation of the best solutions on 10k simulations, descending')


    return solutions


if __name__ == '__main__':
    # Prints out the all hyperparameters for CMA
    for k in cma.CMAOptions():
        print(k, cma.CMAOptions()[k])