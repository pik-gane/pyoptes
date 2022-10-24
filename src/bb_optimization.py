import os.path

from pyoptes.optimization.budget_allocation import target_function as f

from pyoptes import bo_cma, bo_pyGPGO, bo_neural_process

from pyoptes import choose_sentinels, baseline
from pyoptes import map_low_dim_x_to_high_dim, create_test_strategy_prior
from pyoptes import save_hyperparameters, save_results, plot_prior, create_graph, save_raw_data
from pyoptes import plot_time_for_optimization, plot_optimizer_history, evaluate_prior
from pyoptes import compute_average_otf_and_stderr, softmax
from pyoptes import rms_tia, percentile_tia, mean_tia

import argparse
import numpy as np
from tqdm import tqdm
from time import time
import datetime


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("optimizer", choices=['cma', 'gpgo', 'np'],
                        help="Choose the optimizer to run on the SI-model. Choose between CMA-ES, "
                             "Gaussian Process (GP) and Neural Process (NP).")
    parser.add_argument("name_experiment",
                        help="The name of the folder where the results of the optimizer run are saved to.")

    parser.add_argument("--sentinels", type=int, default=1040,
                        help="Set the number of nodes that are used. Has to be smaller than or equal to n_nodes. "
                             "Default is 120 nodes.")
    parser.add_argument("--n_nodes", type=int, default=1040, choices=[120, 1040, 57590],
                        help="Si-simulation parameter. "
                             "Defines the number of nodes used by the SI-model to create a graph. "
                             "Default value is 120 nodes.")
    parser.add_argument('--n_runs', type=int, default=100,
                        help='The number of times the optimizer is run. Results are then averaged over all runs.'
                             'Default is 100 runs.')
    parser.add_argument('--n_runs_start', type=int, default=0,
                        help='')

    parser.add_argument("--max_iterations", type=int, default=50,
                        help="Optimizer parameter. The maximum number of iterations the algorithms run.")

    # ------------------ GPGO hyperparameters ------------------
    parser.add_argument('--acquisition_function', default='EI',
                        choices=['EI', 'PI', 'UCB', 'Entropy', 'tEI'],
                        help='GPGO optimizer parameter. Defines the acquisition function that is used by GPGO.')
    parser.add_argument('--use_prior', type=bool, default=True,
                        help='GPGO optimizer parameter. Sets whether the surrogate function is fitted with priors '
                             'created by heuristics or by sampling random point. Only works when n_nodes and sentinels'
                             'are the same size. Default is True.')
    parser.add_argument('--prior_mixed_strategies', type=bool, default=False,
                        help='GPGO optimizer parameter. '
                             'Sets whether to use test strategies that mix highest degrees and capacities in the prior.'
                             'If set to no the prior has the same shape for all network sizes.')
    parser.add_argument('--prior_only_baseline', type=bool, default=False,
                        help='GPGO optimizer parameter. Sets whether to use only the baseline strategy in the prior.'
                             'If true the prior consists of only one item.')

    # ------------------ Neural Process hyperparameters ------------------
    parser.add_argument('--r_dim', type=int, default=50, help='')
    parser.add_argument('--z_dim', type=int, default=50, help='')
    parser.add_argument('--h_dim', type=int, default=50, help='')
    parser.add_argument('--num_target', type=int, default=3,
                        help='The context and target size together must not exceed the number '
                             'of the budgets in the prior.')
    parser.add_argument('--num_context', type=int, default=3,
                        help='The context and target size together must not exceed the number '
                             'of the budgets in the prior.')
    parser.add_argument('--z_sample_size', type=int, default=10,
                        help='Sets how many samples are drawn from the posterior distribution '
                             'of the latent variables before averaging.')

    parser.add_argument('--epochs', type=int, default=30,
                        help='GPGO optimizer parameter. Sets the number of epochs of the neural process.')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='GPGO optimizer parameter. Sets the batch size of the neural process.')

    # ------------------ CMA-ES hyperparameters ------------------
    parser.add_argument('--popsize', type=int, default=9,
                        help='CMA-ES optimizer parameter. Defines the size of the population each iteration.'
                             'CMA default is "4+int(3*log(n_nodes))" '
                             '-> 18 of 120, 24 for 1040, 36 for 57590.'
                             'Is set to 9 for performance reasons.')
    parser.add_argument('--scale_sigma', type=float, default=0.25,
                        help='CMA-ES optimizer parameter. Defines the scaling of the standard deviation. '
                             'Default is a standard deviation of 0.25 of the total budget.')
    parser.add_argument('--cma_initial_population', default='uniform', choices=['uniform', 'degree', 'capacity'],
                        help='CMA-ES optimizer parameter. Sets which test strategy in the prior is used as the initial '
                             'population for cma.')

    # ------------------ SI-simulation hyperparameters ------------------
    parser.add_argument("--statistic", choices=['mean', 'rms', '95perc'], default='rms',
                        help="Choose the statistic to be used by the target function. "
                             "Choose between mean, rms (root-mean-square) or 95perc (95th-percentile).")
    parser.add_argument("--n_simulations", type=int, default=1000,
                        help="Si-simulation parameter. Sets the number of runs the for the SI-model. "
                             "Higher values of n_simulations lower the variance of the output of the simulation. "
                             "Default value is 1000.")
    parser.add_argument('--graph_type', choices=['waxman', 'ba', 'syn'], default='syn',
                        help='Si-simulation parameter. Set the type of graph the simulation uses.'
                             ' Either Waxman,Synthetic or Barabasi-Albert (ba) can be used. Default is Synthetic.')
    parser.add_argument('--scale_total_budget', type=int, default=1, choices=[1, 4, 12],
                        help="SI-simulation parameter. Scales the total budget for SI-model. Default is 1.")
    parser.add_argument('--parallel', type=bool, default=True,
                        help='Si-simulation parameter. Sets whether multiple simulations run are to be done in parallel'
                             'or sequentially. Default is set to parallel computation.')
    parser.add_argument("--num_cpu_cores", type=int, default=32,
                        help='Si-simulation parameter. Defines the number of cpus to be used for the simulation '
                             'parallelization. If more cpus are chosen than available, the max available are selected.'
                             '-1 selects all available cpus. Default are 32 cpus.')

    parser.add_argument('--delta_t_symptoms', type=int, default=60,
                        help='Si-simulation parameter.. Sets the time (in days) after which an infection is detected'
                             ' automatically. Default is 60 days')
    parser.add_argument('--p_infection_by_transmission', type=float, default=0.5,
                        help='Si-simulation parameter. The probability of how likely a trade animal '
                             'infects other animals. Default is 0.5.')
    parser.add_argument('--expected_time_of_first_infection', type=int, default=30,
                        help='Si-simulation parameter. '
                             'The expected time (in days) after which the first infection occurs. ')

    # ------------------ utility hyperparameters ------------------
    parser.add_argument('--mode_choose_sentinels', choices=['degree', 'capacity', 'transmission'], default='degree',
                        help='Sets the mode of how sentinels are chosen. ')
    parser.add_argument('--save_test_strategies', type=bool, default='',
                        help='Sets whether to save the test strategies that are evaluate in the optimization.')
    parser.add_argument('--plot_prior', type=bool, default='',
                        help='')
    parser.add_argument("--log_level", type=int, default=3, choices=range(1, 11), metavar="[1-10]",
                        help="Optimizer parameter. Only effects SMAC and GPGO. Sets how often log messages appear. "
                             "Lower values mean more messages.")
    parser.add_argument('--path_plot', default='../data/blackbox_learning/results/',
                        help="Optimizer parameter. Location where all the individual results"
                             " of the optimizers are saved to. "
                             "Default location is 'pyoptes/optimization/budget_allocation/blackbox_learning/plots/'")
    parser.add_argument('--path_networks', default='../../networks/data',
                        help='Location where the networks are saved to. '
                             'Path on cluster. /p/projects/ou/labs/gane/optes/mcmc_100nets/data'
                             '/p/projects/ou/labs/gane/optes/mcmc_100nets/data/')

    args = parser.parse_args()

    # prepare the directory for the plots, hyperparameters and results
    path_experiment = os.path.join(args.path_plot, args.name_experiment)
    if not os.path.exists(path_experiment):
        os.makedirs(path_experiment)

    ###################################################################################################################
    # prepare hyperparameters
    ###################################################################################################################

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

    # map the descriptor for the intial population of cma to an index of the prior
    map_cma_initial_population = {'uniform': 0, 'degree': 1, 'capacity': 2}
    cma_initial_population = map_cma_initial_population[args.cma_initial_population]

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
                                                        'n_runs_start': args.n_runs_start,
                                                        },
                         'optimizer_hyperparameters': {'optimizer': args.optimizer,
                                                       'max_iterations': args.max_iterations,
                                                       }}

    if args.optimizer == 'cma':
        experiment_params['optimizer_hyperparameters']['cma_sigma'] = cma_sigma
        experiment_params['optimizer_hyperparameters']['popsize'] = args.popsize
        experiment_params['optimizer_hyperparameters']['cma_initial_population'] = args.cma_initial_population
    elif args.optimizer == 'gpgo':
        experiment_params['optimizer_hyperparameters']['use_prior'] = args.use_prior
        experiment_params['optimizer_hyperparameters']['acquisition_function'] = acquisition_function
        experiment_params['optimizer_hyperparameters']['prior_mixed_strategies'] = args.prior_mixed_strategies
        experiment_params['optimizer_hyperparameters']['prior_only_baseline'] = args.prior_only_baseline
    elif args.optimizer == 'np':
        experiment_params['optimizer_hyperparameters']['acquisition_function'] = acquisition_function
        experiment_params['optimizer_hyperparameters']['prior_mixed_strategies'] = args.prior_mixed_strategies
        experiment_params['optimizer_hyperparameters']['prior_only_baseline'] = args.prior_only_baseline
        experiment_params['optimizer_hyperparameters']['epochs'] = args.epochs
        experiment_params['optimizer_hyperparameters']['batch_size'] = args.batch_size
        experiment_params['optimizer_hyperparameters']['r_dim'] = args.r_dim
        experiment_params['optimizer_hyperparameters']['z_dim'] = args.z_dim
        experiment_params['optimizer_hyperparameters']['h_dim'] = args.h_dim
        experiment_params['optimizer_hyperparameters']['num_context'] = args.num_context
        experiment_params['optimizer_hyperparameters']['num_target'] = args.num_target
        experiment_params['optimizer_hyperparameters']['z_sample_size'] = args.z_sample_size
    else:
        raise ValueError('Optimizer not supported')

    save_hyperparameters(hyperparameters=experiment_params, base_path=path_experiment)

    ####################################################################################################################
    # start optimization
    ####################################################################################################################

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

    # time for the optimization in each iteration on each network (in minutes, not cumulative)
    list_time_for_optimization = []
    list_time_acquisition_optimization = []
    list_time_update_surrogate = []

    time_start = time()  # start reference for all n optimizer runs

    for n in range(args.n_runs_start, args.n_runs_start + args.n_runs):
        # create a folder to save the results of the individual optimization run
        path_sub_experiment = os.path.join(path_experiment, 'individual', f'{n}')
        if not os.path.exists(path_sub_experiment):
            os.makedirs(path_sub_experiment)

        print(f'Run {n+1} of {args.n_runs+args.n_runs_start},'
              f' start time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

        #
        transmissions, capacities, degrees = create_graph(n=n, graph_type=args.graph_type,
                                                          n_nodes=args.n_nodes, base_path=args.path_networks)

        # initialize the si-simulation
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

        for i, p in tqdm(enumerate(prior), leave=False, total=len(prior)):

            p = map_low_dim_x_to_high_dim(x=p,
                                          number_of_nodes=args.n_nodes,
                                          node_indices=prior_node_indices[i])

            m, stderr = f.evaluate(budget_allocation=p,
                                   n_simulations=args.n_simulations,
                                   parallel=args.parallel,
                                   num_cpu_cores=args.num_cpu_cores,
                                   statistic=statistic)
            list_prior_tf.append(m)
            list_prior_stderr.append(stderr)
        list_all_prior_tf.append(list_prior_tf)
        list_all_prior_stderr.append(list_prior_stderr)

        list_prior.append(prior)

        # save a description of what each strategy is
        with open(os.path.join(path_experiment, f'prior_parameter_{args.n_nodes}_nodes.txt'), 'w') as fi:
            fi.write(prior_parameter)

        # reduce the dimension of the input space by choosing to only allocate the budget between nodes with the highest
        # degrees. The function return the indices of these nodes
        # The indices correspond to the first item of the prior
        node_attributes = [degrees, capacities, transmissions]
        node_indices = choose_sentinels(node_attributes=node_attributes,
                                        sentinels=args.sentinels,
                                        mode=args.mode_choose_sentinels)

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
        save_test_strategies_path = os.path.join(path_sub_experiment, 'raw_data/test_strategies_history')
        if not os.path.exists(save_test_strategies_path) and args.save_test_strategies:
            os.makedirs(save_test_strategies_path)
        optimizer_kwargs = {'n_nodes': args.n_nodes, 'node_indices': node_indices, 'eval_function': f.evaluate,
                            'n_simulations': args.n_simulations, 'statistic': statistic, 'total_budget': total_budget,
                            'max_iterations': args.max_iterations, 'parallel': args.parallel,
                            'num_cpu_cores': args.num_cpu_cores, 'save_test_strategies': args.save_test_strategies,
                            'save_test_strategies_path': save_test_strategies_path}

        t0 = time()  # time for an individual optimization run
        if args.optimizer == 'cma':

            # CMA-ES can take only an initial population of one. For this the uniform baseline is used
            # TODO maybe change to/test with highest degree baseline ?
            optimizer_kwargs['initial_population'] = prior[cma_initial_population]
            optimizer_kwargs['bounds'] = bounds
            optimizer_kwargs['sigma'] = cma_sigma
            optimizer_kwargs['popsize'] = args.popsize

            # optimizers return the best test strategy, a history of the best solutions during a run (with stderror) and
            # the time it took to run the optimizer
            best_test_strategy, best_solution_history, stderr_history, \
            time_for_optimization = \
                bo_cma(**optimizer_kwargs)

        elif args.optimizer == 'gpgo':

            optimizer_kwargs['prior'] = prior
            optimizer_kwargs['prior_y'] = list_prior_tf
            optimizer_kwargs['prior_stderr'] = list_prior_stderr
            optimizer_kwargs['acquisition_function'] = acquisition_function
            optimizer_kwargs['use_prior'] = args.use_prior

            # optimizers return the best test strategy, a history of the best solutions during a run (with stderror) and
            # the time it took to run the optimizer
            best_test_strategy, best_solution_history, stderr_history, \
            time_for_optimization, time_acquisition_optimization, time_update_surrogate = \
                bo_pyGPGO(**optimizer_kwargs)

        elif args.optimizer == 'np':

            # # check validity of neural process hyperparameters
            if args.num_context + args.num_target > len(prior):
                raise ValueError('The context and target size together must not exceed the number of the budgets in the prior.'
                                 f'Got {args.num_context} + {args.num_target} > {len(prior)}')

            optimizer_kwargs['prior'] = prior
            optimizer_kwargs['prior_y'] = list_prior_tf
            optimizer_kwargs['prior_stderr'] = list_prior_stderr
            optimizer_kwargs['acquisition_function'] = acquisition_function
            optimizer_kwargs['epochs'] = args.epochs
            optimizer_kwargs['batch_size'] = args.batch_size
            optimizer_kwargs['r_dim'] = args.r_dim
            optimizer_kwargs['z_dim'] = args.z_dim
            optimizer_kwargs['h_dim'] = args.h_dim
            optimizer_kwargs['num_context'] = args.num_context
            optimizer_kwargs['num_target'] = args.num_target
            optimizer_kwargs['z_sample_size'] = args.z_sample_size

            # optimizers return the best test strategy, a history of the best solutions during a run (with stderror) and
            # the time it took to run the optimizer
            best_test_strategy, best_solution_history, stderr_history,\
            time_for_optimization, time_acquisition_optimization, time_update_surrogate = \
                bo_neural_process(**optimizer_kwargs)

        else:
            raise ValueError('Optimizer not supported')

        print('------------------------------------------------------')

        # plot and save to disk the results of the individual optimization runs
        plot_optimizer_history(optimizer_history=best_solution_history, stderr_history=stderr_history,
                               baseline_mean=baseline_mean, baseline_stderr=baseline_stderr,
                               n_nodes=args.n_nodes, sentinels=args.sentinels,
                               path_experiment=path_sub_experiment, optimizer=args.optimizer)

        plot_time_for_optimization(time_for_optimization=time_for_optimization,
                                   path_experiment=path_sub_experiment,
                                   optimizer=args.optimizer)

        if args.optimizer == 'gpgo' or args.optimizer == 'np':
            plot_time_for_optimization(time_for_optimization=time_acquisition_optimization,
                                       path_experiment=path_sub_experiment,
                                       optimizer=args.optimizer,
                                       file_name='time_for_acquisition_optimization.png',
                                       title='Time for acquisition function optimization')
            plot_time_for_optimization(time_for_optimization=time_update_surrogate,
                                       path_experiment=path_sub_experiment,
                                       optimizer=args.optimizer,
                                       file_name='time_for_surrogate_update.png',
                                       title='Time for surrogate function update')

        # get the best strategy from the history of the optimizer
        index = np.argmin(best_solution_history)
        eval_best_test_strategy = best_solution_history[index]
        best_test_strategy_stderr = stderr_history[index]

        # a ratio of less than 100% means that the optimizer did not find a strategy that is better than the baseline
        # negative value are undesirable and indicate that the optimizer did not find a strategy that is better
        ratio_otf = 100 - 100*(eval_best_test_strategy / baseline_mean)

        output = f'\nTime for optimization (in hours): {(time() - t0) / 60}' \
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
        # times for acquisition and surrogate update are only available for gpgo and np
        if args.optimizer == 'gpgo' or args.optimizer == 'np':
            list_time_acquisition_optimization.append(time_acquisition_optimization)
            list_time_update_surrogate.append(time_update_surrogate)

    # save the results of the experiment
    raw_data_path = os.path.join(path_experiment, 'raw_data', str(args.n_runs_start))

    # save the raw data of the optimization runs
    kwargs_save_raw_data = {'list_best_otf': list_best_otf,
                            'list_best_otf_stderr': list_best_otf_stderr,
                            'list_baseline_otf': list_baseline_otf,
                            'list_baseline_otf_stderr': list_baseline_otf_stderr,
                            'list_ratio_otf': list_ratio_otf,
                            'list_best_solution_history': list_best_solution_history,
                            'list_stderr_history': list_stderr_history,
                            'list_time_for_optimization': list_time_for_optimization,
                            'list_all_prior_tf': list_all_prior_tf,
                            'list_all_prior_stderr': list_all_prior_stderr,
                            'path_experiment': raw_data_path}
    # times for acquisition and surrogate update are only available for gpgo and np
    if args.optimizer == 'gpgo' or args.optimizer == 'np':
        kwargs_save_raw_data['list_time_acquisition_optimization'] = list_time_acquisition_optimization
        kwargs_save_raw_data['list_time_update_surrogate'] = list_time_update_surrogate
    save_raw_data(**kwargs_save_raw_data)

    # create a .done file in the sub path to indicate the run is finished
    with open(os.path.join(raw_data_path, '.done'), 'w') as done_file:
        done_file.write(f'Individual run {n} finished')

    print(f'Optimization end: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')

    # ------------------------------------------------------------
    # postprocessing of all runs
    # ------------------------------------------------------------
    # compute the averages over all optimization runs of the prior, the optimizer, the baseline and their standard error

    # compute the average OTFs, baseline and their standard errors
    average_best_otf, average_best_otf_stderr = compute_average_otf_and_stderr(list_otf=list_best_otf,
                                                                               list_stderr=list_best_otf_stderr,
                                                                               n_runs=args.n_runs)

    average_baseline, average_baseline_stderr = compute_average_otf_and_stderr(list_otf=list_baseline_otf,
                                                                               list_stderr=list_baseline_otf_stderr,
                                                                               n_runs=args.n_runs)

    average_ratio_otf = np.mean(list_ratio_otf)

    output = f'Results averaged over {args.n_runs} optimizer runs' \
             f'\naverage ratio otf to baseline: {average_ratio_otf}' \
             f'\naverage baseline and stderr: {average_baseline}, {average_baseline_stderr}' \
             f'\naverage best strategy OTF and stderr: {average_best_otf}, {average_best_otf_stderr}' \
             f'\nTime for optimization (in hours): {(time() - time_start) / 3600}'

    save_results(best_test_strategy=None,
                 save_test_strategy=False,
                 path_experiment=os.path.join(path_experiment, 'raw_data', str(args.n_runs_start)),
                 output=output)
    print(output)
