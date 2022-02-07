import cma


def bo_cma(objective_function, initial_population,
           sigma=0.2,
           max_iterations=1000,
           n_simulations=1000,
           indices=[],
           true_size_x=120):
    """

    @param n_simulations:
    @param indices:
    @param true_size_x:
    @param objective_function:
    @param initial_population:
    @param sigma:
    @param max_iterations:
    @return:
    """
    # TODO look into param "bounds" to set upper and lower bounds of solutions
    # ea = cma.CMAEvolutionStrategy(initial_population, sigma, inopts={'maxiter': max_iterations, 'verbose': -8})
    # ea.optimize(objective_function)
    # solutions = ea.pop_sorted

    ea = cma.fmin(objective_function, initial_population, sigma0=sigma, options={'maxiter': max_iterations,
                                                                                 'verbose': -8,
                                                                                 'verb_plot': 0},
                  args=(n_simulations, indices, true_size_x))

    # logger = ea[-1].load()
    # logger.plot_all()
    # print(solutions)
    # cma.s.figsave('f')
    # cma.plot('outcmaes')
    # print('\nEvaluation of the best solutions on 10k simulations, descending')
    # for s in solutions:
    #     print(objective_function(s, n_simulations=10000))

    return 0
    # return solutions


if __name__ == '__main__':
    # Prints out the all hyperparameters for CMA
    for k in cma.CMAOptions():
        print(k, cma.CMAOptions()[k])