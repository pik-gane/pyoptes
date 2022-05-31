import os.path

from pyoptes.optimization.budget_allocation import target_function as f

from pyoptes import bo_cma, bo_pyGPGO

from pyoptes import choose_high_degree_nodes, baseline
from pyoptes import map_low_dim_x_to_high_dim, create_test_strategy_prior
from pyoptes import save_hyperparameters, save_results, plot_prior, create_graphs, save_raw_data
from pyoptes import plot_time_for_optimization, plot_optimizer_history, evaluate_prior
from pyoptes import compute_average_otf_and_stderr

import argparse
import numpy as np
from tqdm import tqdm
from scipy.stats.mstats import mjci
from time import time
import datetime


def rms_tia(n_infected_animals):
    values = n_infected_animals**2
    estimate = np.sqrt(np.mean(values, axis=0))
    stderr = np.std(values, ddof=1, axis=0) / np.sqrt(values.shape[0])
    stderr = stderr/(2*estimate)
    return estimate, stderr


def mean_tia(n_infected_animals):
    estimate = np.mean(n_infected_animals, axis=0)
    stderr = np.std(n_infected_animals, ddof=1, axis=0) / np.sqrt(n_infected_animals.shape[0])
    return estimate, stderr


def percentile_tia(n_infected_animals):
    estimate = np.percentile(n_infected_animals, 95, axis=0)
    stderr = mjci(n_infected_animals, prob=[0.95], axis=0)[0]
    return estimate, stderr

# TODO add this
def share_detected(unused_n_infected_animals):
    return model.detection_was_by_test.true

# TODO run GPGO-experiments without the prior but similar number of function evals
# prior -> 39 evals for fit + 50 evals for optim
# no prior -> 3 random samples for fit + 86 evals for optim
# only baseline as prior -> 1 sample for fit + 89 evals for optim
# only baseline + highest degree/cap -> 7 for fit (regardless of network size) + 82 evals for optim
# or just always use 50 evals for optim, for better comparison


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("optimizer", choices=['cma', 'gpgo'],
                        help="Choose the optimizer to run on the SI-model. Choose between CMA-ES and GPGO")
    parser.add_argument("name_experiment",
                        help="The name of the folder where the results of the optimizer run are saved to.")

    parser.add_argument("--sentinels", type=int, default=120,
                        help="Set the number of nodes that are used. Has to be smaller than or equal to n_nodes. "
                             "Default is 120 nodes.")
    parser.add_argument("--n_nodes", type=int, default=120, choices=[120, 1040, 57590],
                        help="Si-simulation parameter. "
                             "Defines the number of nodes used by the SI-model to create a graph. "
                             "Default value is 120 nodes.")
    parser.add_argument('--n_runs', type=int, default=100,
                        help='The number of times the optimizer is run. Results are then averaged over all runs.'
                             'Default is 100 runs.')

    parser.add_argument("--max_iterations", type=int, default=50,
                        help="Optimizer parameter. The maximum number of iterations the algorithms run.")

    parser.add_argument('--acquisition_function', default='EI',
                        choices=['EI', 'PI', 'UCB', 'Entropy', 'tEI'],
                        help='GPGO optimizer parameter. Defines the acquisition function that is used by GPGO.')
    parser.add_argument('--use_prior', type=bool, default=True,
                        help='GPGO optimizer parameter. Sets whether the surrogate function is fitted with priors '
                             'created by heuristics or by sampling random point. Only works when n_nodes and sentinels'
                             'are the same size. Default is True.')
    parser.add_argument('--prior_mixed_strategies', type=bool, default=True,
                        help='GPGO optimizer parameter. '
                             'Sets whether to use test strategies that mix highest degrees and capacities in the prior.'
                             'If set to no the prior has the same shape for all network sizes.')
    parser.add_argument('--prior_only_baseline', type=bool, default=False,
                        help='GPGO optimizer parameter. Sets whether to use only the baseline strategy in the prior.'
                             'If true the prior consists of only one item.')

    parser.add_argument('--popsize', type=int, default=18,
                        help='CMA-ES optimizer parameter. Defines the size of the population each iteration.'
                             'CMA default is "4+int(3*log(n_nodes))" '
                             '-> 18 of 120, 24 for 1040, 36 for 57590.')
    parser.add_argument('--scale_sigma', type=float, default=0.25,
                        help='CMA-ES optimizer parameter. Defines the scaling of the standard deviation. '
                             'Default is a standard deviation of 0.25 of the total budget.')

    parser.add_argument("--statistic", choices=['mean', 'rms', '95perc'], default='rms',
                        help="Choose the statistic to be used by the target function. "
                             "Choose between mean, rms (root-mean-square) or 95perc (95th-percentile).")
    parser.add_argument("--n_simulations", type=int, default=10000,
                        help="Si-simulation parameter. Sets the number of runs the for the SI-model. "
                             "Higher values of n_simulations lower the variance of the output of the simulation. "
                             "Default value is 1000.")
    parser.add_argument('--graph_type', choices=['waxman', 'ba', 'syn'], default='ba',
                        help='Si-simulation parameter. Set the type of graph the simulation uses.'
                             ' Either Waxman or Barabasi-Albert (ba) can be used. Default is Barabasi-Albert.')
    parser.add_argument('--delta_t_symptoms', type=int, default=60,
                        help='Si-simulation parameter.. Sets the time (in days) after which an infection is detected'
                             ' automatically. Default is 60 days')
    parser.add_argument('--p_infection_by_transmission', type=float, default=0.5,
                        help='Si-simulation parameter. The probability of how likely a trade animal '
                             'infects other animals. Default is 0.5.')
    parser.add_argument('--expected_time_of_first_infection', type=int, default=30,
                        help='Si-simulation parameter. The expected time (in days) after which the first infection occurs. ')
    parser.add_argument('--parallel', type=bool, default=True,
                        help='Si-simulation parameter. Sets whether multiple simulations run are to be done in parallel'
                             'or sequentially. Default is set to parallel computation.')
    parser.add_argument("--num_cpu_cores", type=int, default=32,
                        help='Si-simulation parameter. Defines the number of cpus to be used for the simulation '
                             'parallelization. If more cpus are chosen than available, the max available are selected.'
                             '-1 selects all available cpus. Default are 14 cpus.')
    parser.add_argument('--scale_total_budget', type=int, default=1, choices=[1, 4, 12],
                        help="SI-simulation parameter. Scales the total budget for SI-model. Default is 1.")

    parser.add_argument('--plot_prior', type=bool, default='',
                        help='')
    parser.add_argument("--log_level", type=int, default=3, choices=range(1, 11), metavar="[1-10]",
                        help="Optimizer parameter. Only effects SMAC and GPGO. Sets how often log messages appear. "
                             "Lower values mean more messages.")
    parser.add_argument('--path_plot', default='pyoptes/optimization/budget_allocation/blackbox_learning/plots/',
                        help="Optimizer parameter. Location where all the individual results"
                             " of the optimizers are saved to. "
                             "Default location is 'pyoptes/optimization/budget_allocation/blackbox_learning/plots/'")
    parser.add_argument('--path_networks', default='../data',
                        help='Location where the networks are saved to. '
                             'Path on cluster. /p/projects/ou/labs/gane/optes/mcmc_100nets/data'
                             '/p/projects/ou/labs/gane/optes/mcmc_100nets/data/')
    args = parser.parse_args()

    # prepare the directory for the plots, hyperparameters and results
    path_experiment = os.path.join(args.path_plot, args.name_experiment)
    if not os.path.exists(path_experiment):
        os.makedirs(path_experiment)

    # TODO change hyperparameter of acquisition function
    # TODO test different acquisition functions
    # map acquisition function string to one useable by pyGPGO. This is just to keep command-line args short
    af = {'EI': 'ExpectedImprovement', 'PI': 'ProbabilityImprovement', 'UCB': 'UCB',
          'Entropy': 'Entropy', 'tEI': 'tExpectedImprovement'}
    acquisition_function = af[args.acquisition_function]

    # define function to average the results of the simulation
    if args.statistic == 'mean':
        statistic = mean_tia
    elif args.statistic == 'rms':
        statistic = rms_tia
    elif args.statistic == '95perc':
        statistic = percentile_tia
    else:
        raise ValueError('Statistic not supported')

    # The total budget is a function of the number of nodes in the network and
    total_budget = args.scale_total_budget * args.n_nodes  # i.e., on average, nodes will do one test per year
    # define the first constraint, the boundaries of x_i
    bounds = [0, total_budget]

    # for CMA-ES, sigma is set as 0.25 of the total budget
    cma_sigma = args.scale_sigma * total_budget

    # save SI-model and optimizer parameters as .json-file
    experiment_params = {'simulation_hyperparameters': {'total_budget': total_budget,
                                                        'n_nodes': args.n_nodes,
                                                        'graph': args.graph_type,
                                                        'sentinels': args.sentinels,
                                                        'n_simulations': args.n_simulations,
                                                        'delta_t_symptoms': args.delta_t_symptoms,
                                                        'p_infection_by_transmission': args.p_infection_by_transmission,
                                                        'expected_time_of_first_infection': args.expected_time_of_first_infection,
                                                        'n_runs': args.n_runs,
                                                        'statistic': args.statistic,
                                                        },
                         'optimizer_hyperparameters': {'optimizer': args.optimizer,
                                                       'max_iterations': args.max_iterations,
                                                       }}

    # TODO move network creation into for loop
    # TODO use Synthetic/Ansari-networks instead BA/waxman with budget N, 1040 nodes, RMS
    # creates a list of n_runs networks (either waxman or barabasi-albert)
    network_list = create_graphs(args.n_runs, args.graph_type, args.n_nodes, args.path_networks)

    if args.optimizer == 'cma':
        experiment_params['optimizer_hyperparameters']['cma_sigma'] = cma_sigma
        experiment_params['optimizer_hyperparameters']['popsize'] = args.popsize
    elif args.optimizer == 'gpgo':
        experiment_params['optimizer_hyperparameters']['use_prior'] = args.use_prior
        experiment_params['optimizer_hyperparameters']['acquisition_function'] = acquisition_function
        experiment_params['optimizer_hyperparameters']['prior_mixed_strategies'] = args.prior_mixed_strategies
        experiment_params['optimizer_hyperparameters']['prior_only_baseline'] = args.prior_only_baseline
    else:
        raise ValueError('Optimizer not supported')

    save_hyperparameters(experiment_params, path_experiment)

    # lists for result aggregations
    list_prior = []

    list_best_otf = []  # best optimizer function value on each network and corresponding standard error
    list_best_otf_stderr = []
    list_baseline_otf = []  # baseline  function value on each network and corresponding standard error
    list_baseline_otf_stderr = []

    list_ratio_otf = []     # ratio of best optimizer function value to baseline function value on each network
    list_best_solution_history = []
    list_stderr_history = []

    list_all_prior_tf = []
    list_all_prior_stderr = []

    list_time_for_optimization = []

    time_start = time()
    for n, network in enumerate(network_list[:args.n_runs]):
        print(f'Run {n + 1} of {args.n_runs},'
              f' start time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

        # create a folder to save the results of the individual optimization run
        path_sub_experiment = os.path.join(path_experiment, 'individual', f'{n}')
        if not os.path.exists(path_sub_experiment):
            os.makedirs(path_sub_experiment)

        # unpack the properties of the network
        transmissions, capacities, degrees = network

        f.prepare(n_nodes=args.n_nodes,
                  capacity_distribution=capacities,
                  pre_transmissions=transmissions,
                  p_infection_by_transmission=args.p_infection_by_transmission,
                  delta_t_symptoms=args.delta_t_symptoms,
                  expected_time_of_first_infection=args.expected_time_of_first_infection,
                  static_network=None,
                  use_real_data=False)

        # create a list of test strategies based on different heuristics
        prior, prior_node_indices, prior_parameter = \
            create_test_strategy_prior(n_nodes=args.n_nodes,
                                       node_degrees=degrees,
                                       node_capacities=capacities,
                                       total_budget=total_budget,
                                       sentinels=args.sentinels,
                                       mixed_strategies=args.prior_mixed_strategies,
                                       only_baseline=args.prior_only_baseline)

        # evaluate the strategies in the prior
        list_prior_tf = []
        list_prior_stderr = []
        for p in tqdm(prior, leave=False):
            m, stderr = f.evaluate(budget_allocation=p,
                                   n_simulations=args.n_simulations,
                                   parallel=args.parallel,
                                   num_cpu_cores=args.num_cpu_cores,
                                   statistic=statistic)
            list_prior_tf.append(m)
            list_prior_stderr.append(stderr)
        list_all_prior_tf.append(list_prior_tf)
        list_all_prior_stderr.append(list_prior_stderr)

        # list_prior is only needed if the objective function values of the strategies in the prior
        # are to be plotted
        list_prior.append(prior)

        # save a description of what each strategy is
        with open(os.path.join(path_experiment, f'prior_parameter_{args.n_nodes}_nodes.txt'), 'w') as fi:
            fi.write(prior_parameter)

        # reduce the dimension of the input space by choosing to only allocate the budget between nodes with the highest
        # degrees. The function return the indices of these nodes
        # The indices correspond to the first item of the prior
        node_indices = choose_high_degree_nodes(degrees, args.n_nodes, args.sentinels)

        # compute the baseline, i.e., the expected value of the objective function for a uniform distribution of the
        # budget over all nodes (regardless of the number of sentinels)
        baseline_mean, baseline_stderr, x_baseline = baseline(total_budget=total_budget,
                                                              eval_function=f.evaluate,
                                                              n_nodes=args.n_nodes,
                                                              parallel=args.parallel,
                                                              num_cpu_cores=args.num_cpu_cores,
                                                              statistic=statistic)

        # ----------------------------------------
        # start the chosen optimizer
        # shared optimizer parameters
        optimizer_kwargs = {'n_nodes': args.n_nodes, 'node_indices': node_indices, 'eval_function': f.evaluate,
                            'n_simulations': args.n_simulations, 'statistic': statistic, 'total_budget': total_budget,
                            'max_iterations': args.max_iterations,
                            'parallel': args.parallel, 'num_cpu_cores': args.num_cpu_cores}

        t0 = time()
        if args.optimizer == 'cma':

            optimizer_kwargs['initial_population'] = prior[0]   # CMA-ES can take only an initial population of one
            optimizer_kwargs['bounds'] = bounds
            optimizer_kwargs['sigma'] = cma_sigma
            optimizer_kwargs['popsize'] = args.popsize
            optimizer_kwargs['log_path'] = path_sub_experiment


            best_test_strategy, best_solution_history, stderr_history, time_for_optimization = \
                bo_cma(**optimizer_kwargs)

        elif args.optimizer == 'gpgo':

            optimizer_kwargs['prior'] = prior
            optimizer_kwargs['acquisition_function'] = acquisition_function
            optimizer_kwargs['use_prior'] = args.use_prior

            best_test_strategy, best_solution_history, stderr_history, time_for_optimization, = \
                bo_pyGPGO(**optimizer_kwargs)

        else:
            raise ValueError('Optimizer not supported')

        print('------------------------------------------------------')

        # plot and save to disk the results of the individual optimization runs
        plot_optimizer_history(best_solution_history, stderr_history,
                               baseline_mean, baseline_stderr,
                               args.n_nodes, args.sentinels,
                               path_sub_experiment, args.optimizer)

        plot_time_for_optimization(time_for_optimization,
                                   args.n_nodes, args.sentinels,
                                   path_sub_experiment, args.optimizer)

        best_test_strategy = total_budget * np.exp(best_test_strategy) / sum(np.exp(best_test_strategy))
        best_test_strategy = map_low_dim_x_to_high_dim(best_test_strategy, args.n_nodes, node_indices)

        eval_best_test_strategy, best_test_strategy_stderr = f.evaluate(best_test_strategy,
                                                                        n_simulations=args.n_simulations,
                                                                        parallel=args.parallel,
                                                                        num_cpu_cores=args.num_cpu_cores,
                                                                        statistic=statistic)

        ratio_otf = 100 - (eval_best_test_strategy / baseline_mean)

        output = f'\nTime for optimization (in minutes): {(time() - t0) / 60}' \
                 f'\n\nBaseline for uniform budget distribution:  {baseline_mean}' \
                 f'\n Baseline standard-error:  {baseline_stderr}, ' \
                 f'ratio stderr/mean: {baseline_stderr/baseline_mean}' \
                 f'\nBest solutions:' \
                 f'\nObjective value:   {eval_best_test_strategy}' \
                 f'\nStandard error:  {best_test_strategy_stderr},' \
                 f' ratio stderr/mean: {best_test_strategy_stderr/eval_best_test_strategy}' \
                 f'\nRatio OTF: {ratio_otf}'

        #
        save_results(best_test_strategy=best_test_strategy,
                     path_experiment=path_sub_experiment,
                     output=output)

        # save OTFs of baseline and optimizer
        list_best_otf.append(eval_best_test_strategy)
        list_best_otf_stderr.append(best_test_strategy_stderr)
        list_baseline_otf.append(baseline_mean)
        list_baseline_otf_stderr.append(baseline_stderr)

        list_ratio_otf.append(ratio_otf)
        list_best_solution_history.append(best_solution_history)
        list_stderr_history.append(stderr_history)

        list_time_for_optimization.append(time_for_optimization)

    # TODO move the whole postprocessing step and saving of results into seperate step to save memory

    # save the raw data of the optimization runs
    save_raw_data(list_best_otf, list_best_otf_stderr, list_baseline_otf, list_baseline_otf_stderr,
                  list_ratio_otf, list_best_solution_history, list_stderr_history, list_time_for_optimization,
                  list_all_prior_tf, list_all_prior_stderr,
                  path_experiment=path_experiment)
    print(f'Optimization end: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
    # ------------------------------------------------------------
    # postprocessing of all runs
    # ------------------------------------------------------------

    average_prior_tf, average_prior_stderr = compute_average_otf_and_stderr(list_all_prior_tf,
                                                                            list_all_prior_stderr,
                                                                            n_runs=args.n_runs)
    # compute the average OTFs, baseline and their standard errors
    # TODO move into sepearte function
    average_best_otf, average_best_otf_stderr = compute_average_otf_and_stderr(list_best_otf,
                                                                               list_best_otf_stderr,
                                                                               n_runs=args.n_runs)

    average_baseline, average_baseline_stderr = compute_average_otf_and_stderr(list_baseline_otf,
                                                                               list_baseline_otf_stderr,
                                                                               n_runs=args.n_runs)

    average_ratio_otf = np.mean(list_ratio_otf)

    # create an average otf plot
    average_best_solution_history, average_stderr_history = compute_average_otf_and_stderr(list_best_solution_history,
                                                                                           list_stderr_history,
                                                                                           n_runs=args.n_runs)

    plot_optimizer_history(average_best_solution_history,
                           average_stderr_history, # TODO computation of stderr is wrong, check in spreadsheet
                           average_baseline, average_baseline_stderr,
                           args.n_nodes, args.sentinels,
                           path_experiment, args.optimizer,
                           name='_average_plot')

    output = f'Results averaged over {args.n_runs} optimizer runs' \
             f'\naverage ratio otf to baseline: {average_ratio_otf}' \
             f'\naverage baseline and stderr: {average_baseline}, {average_baseline_stderr}' \
             f'\naverage best strategy OTF and stderr: {average_best_otf}, {average_best_otf_stderr}' \
             f'\nTime for optimization (in hours): {(time()-time_start) / 3600}'

    save_results(best_test_strategy=None,
                 save_test_strategy=False,
                 path_experiment=path_experiment,
                 output=output)
    print(output)

    # TODO scatterplot degree /capacity/ incoming transmissions vs strategy

    # TODO this has to be changed, right now the evaluation is only done on one network
    # evaluate each strategy in each prior of n_runs and plot the average objective function value of each strategy
    if args.plot_prior:
        y_prior = []
        print(f'Evaluating prior {args.n_runs} times.')
        for prior in tqdm(list_prior):
            y_prior.append(evaluate_prior(prior,
                                          n_simulations=args.n_simulations,
                                          eval_function=f.evaluate,
                                          parallel=args.parallel,
                                          num_cpu_cores=args.num_cpu_cores,
                                          statistic=statistic))
        y_prior = np.array(y_prior)

        y_prior_mean = np.mean(y_prior[:, :, 0], axis=0)
        # TODO this has to be corrected to the right formula above
        y_prior_stderr = np.mean(y_prior[:, :, 1], axis=0)

        # plot the objective function values of the prior
        plot_prior(path_experiment=path_experiment,
                   n_nodes=args.n_nodes,
                   y_prior_mean=y_prior_mean,
                   y_prior_stderr=y_prior_stderr,
                   n_runs=args.n_runs)
