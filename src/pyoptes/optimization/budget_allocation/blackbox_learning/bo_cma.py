import cma


def bo_cma(objective_function, initial_population,
           sigma=0.2,
           max_iterations=1000):
    """

    @param objective_function:
    @param initial_population:
    @param sigma:
    @param max_iterations:
    @return:
    """
    # TODO look into param "bounds" to set upper and lower bounds of solutions
    ea = cma.CMAEvolutionStrategy(initial_population, sigma, inopts={'maxiter': max_iterations, 'verbose': -8})
    ea.optimize(objective_function)
    solutions = ea.pop_sorted

    for s in solutions:
        print(objective_function(s, n_simulations=10000))

    return solutions


if __name__ == '__main__':
    # Prints out the all hyperparameters for CMA
    for k in cma.CMAOptions():
        print(k, cma.CMAOptions()[k])