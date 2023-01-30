'''
Runs the black-box optimization, saves experiment hyperparameters and results.
'''
import os.path

from pyoptes.optimization.budget_allocation import target_function as f

from pyoptes import bo_cma, bo_pyGPGO, bo_neural_process

from pyoptes import bo_choose_sentinels, bo_baseline
from pyoptes import bo_map_low_dim_x_to_high_dim, bo_create_test_strategy_prior
from pyoptes import bo_save_hyperparameters, bo_save_results, bo_plot_prior, bo_create_graph, bo_save_raw_data
from pyoptes import bo_plot_time_for_optimization, bo_plot_optimizer_history, bo_evaluate_prior
from pyoptes import bo_compute_average_otf_and_stderr, bo_softmax
from pyoptes import bo_rms_tia, bo_percentile_tia, bo_mean_tia

import numpy as np
from tqdm import tqdm
from time import time
import datetime


def bbo_optimization(optimizer: str,
                     name_experiment: str,
                     sentinels: int = 1040,
                     n_nodes: int = 1040,
                     n_runs: int = 100,
                     n_runs_start: int = 0,
                     max_iterations: int = 50,
                     acquisition_function: str = "EI",
                     use_prior: bool = True,
                     prior_mixed_strategies: bool = False,
                     prior_only_baseline: bool = False,
                     r_dim: int = 50,
                     z_dim: int = 50,
                     h_dim: int = 50,
                     num_target: int = 3,
                     num_context: int = 3,
                     z_sample_size: int = 10,
                     epochs: int = 30,
                     batch_size: int = 30,
                     popsize: int = 9,
                     scale_sigma: float = 0.25,
                     cma_initial_population: str = "uniform",
                     statistic_str: str = "mean",
                     n_simulations: int = 1000,
                     graph_type: str = "syn",
                     scale_total_budget: int = 1,
                     parallel: bool = True,
                     num_cpu_cores: int = 32,
                     delta_t_symptoms: int = 60,
                     p_infection_by_transmission: float = 0.5,
                     expected_time_of_first_infection: int = 30,
                     mode_choose_sentinels: str = "degree",
                     save_test_strategies: bool = False,
                     plot_prior: bool = False,
                     path_plot: str = "../data/blackbox_learning/results/",
                     path_networks: str = "../../networks/data"
):

    # prepare the directory for the plots, hyperparameters and results
    path_experiment = os.path.join(path_plot, name_experiment)
    if not os.path.exists(path_experiment):
        os.makedirs(path_experiment)

    ###################################################################################################################
    # prepare hyperparameters
    ###################################################################################################################

    # TODO change the hyperparameters of the acquisition functions
    # map acquisition function string to one useable by pyGPGO. This is just to keep command-line args short
    af = {'EI': 'ExpectedImprovement', 'PI': 'ProbabilityImprovement', 'UCB': 'UCB',
          'Entropy': 'Entropy', 'tEI': 'tExpectedImprovement'}
    acquisition_function = af[acquisition_function]

    # define function to average the results of the simulation
    if statistic_str == 'mean':
        statistic = bo_mean_tia
    elif statistic_str == 'rms':
        statistic = bo_rms_tia
    elif statistic_str == '95perc':
        statistic = bo_percentile_tia
    else:
        raise ValueError('Statistic not supported')

    # The total budget is a function of the number of nodes in the network and
    total_budget = scale_total_budget * n_nodes  # i.e., on average, nodes will do one test per year
    # define the first constraint, the boundaries of x_i
    bounds = [0, total_budget]

    # for CMA-ES, sigma is set as 0.25 of the total budget
    cma_sigma = scale_sigma * total_budget

    # map the descriptor for the initial population of cma to an index of the prior
    map_cma_initial_population = {'uniform': 0, 'degree': 1, 'capacity': 2}
    cma_initial_population = map_cma_initial_population[cma_initial_population]

    # save SI-model and optimizer parameters as .json-file
    experiment_params = {'simulation_hyperparameters': {'total_budget': total_budget,
                                                        'n_nodes': n_nodes,
                                                        'graph': graph_type,
                                                        'sentinels': sentinels,
                                                        'n_simulations': n_simulations,
                                                        'delta_t_symptoms': delta_t_symptoms,
                                                        'p_infection_by_transmission': p_infection_by_transmission,
                                                        'expected_time_of_first_infection': expected_time_of_first_infection,
                                                        'n_runs': n_runs,
                                                        'statistic': statistic_str,
                                                        'n_runs_start': n_runs_start,
                                                        'mode_choose_sentinels': mode_choose_sentinels,
                                                        },
                         'optimizer_hyperparameters': {'optimizer': optimizer,
                                                       'max_iterations': max_iterations,
                                                       }}

    if optimizer == 'cma':
        experiment_params['optimizer_hyperparameters']['cma_sigma'] = cma_sigma
        experiment_params['optimizer_hyperparameters']['popsize'] = popsize
        # TODO saves the value (int) of the dict instead of the key (string)
        experiment_params['optimizer_hyperparameters']['cma_initial_population'] = cma_initial_population
    elif optimizer == 'gpgo':
        experiment_params['optimizer_hyperparameters']['use_prior'] = use_prior
        experiment_params['optimizer_hyperparameters']['acquisition_function'] = acquisition_function
        experiment_params['optimizer_hyperparameters']['prior_mixed_strategies'] = prior_mixed_strategies
        experiment_params['optimizer_hyperparameters']['prior_only_baseline'] = prior_only_baseline
    elif optimizer == 'np':
        experiment_params['optimizer_hyperparameters']['acquisition_function'] = acquisition_function
        experiment_params['optimizer_hyperparameters']['prior_mixed_strategies'] = prior_mixed_strategies
        experiment_params['optimizer_hyperparameters']['prior_only_baseline'] = prior_only_baseline
        experiment_params['optimizer_hyperparameters']['epochs'] = epochs
        experiment_params['optimizer_hyperparameters']['batch_size'] = batch_size
        experiment_params['optimizer_hyperparameters']['r_dim'] = r_dim
        experiment_params['optimizer_hyperparameters']['z_dim'] = z_dim
        experiment_params['optimizer_hyperparameters']['h_dim'] = h_dim
        experiment_params['optimizer_hyperparameters']['num_context'] = num_context
        experiment_params['optimizer_hyperparameters']['num_target'] = num_target
        experiment_params['optimizer_hyperparameters']['z_sample_size'] = z_sample_size
    else:
        raise ValueError('Optimizer not supported')

    bo_save_hyperparameters(hyperparameters=experiment_params, base_path=path_experiment)

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

    for n in range(n_runs_start, n_runs_start + n_runs):
        # create a folder to save the results of the individual optimization run
        path_sub_experiment = os.path.join(path_experiment, 'individual', f'{n}')
        if not os.path.exists(path_sub_experiment):
            os.makedirs(path_sub_experiment)

        print(f'Run {n+1} of {n_runs+n_runs_start},'
              f' start time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

        #
        transmissions, capacities, degrees = bo_create_graph(n=n, graph_type=graph_type,
                                                             n_nodes=n_nodes, base_path=path_networks)

        # initialize the si-simulation
        f.prepare(n_nodes=n_nodes,
                  capacity_distribution=capacities,
                  pre_transmissions=transmissions,
                  p_infection_by_transmission=p_infection_by_transmission,
                  delta_t_symptoms=delta_t_symptoms,
                  expected_time_of_first_infection=expected_time_of_first_infection,
                  static_network=None,
                  use_real_data=False)

        # create a list of test strategies based on different heuristics
        prior, prior_node_indices, prior_parameter = \
            bo_create_test_strategy_prior(n_nodes=n_nodes,
                                          node_degrees=degrees,
                                          node_capacities=capacities,
                                          total_budget=total_budget,
                                          sentinels=sentinels,
                                          mixed_strategies=prior_mixed_strategies,
                                          only_baseline=prior_only_baseline)

        # evaluate the strategies in the prior
        list_prior_tf = []
        list_prior_stderr = []

        for i, p in tqdm(enumerate(prior), leave=False, total=len(prior)):

            p = bo_map_low_dim_x_to_high_dim(x=p,
                                             number_of_nodes=n_nodes,
                                             node_indices=prior_node_indices[i])

            m, stderr = f.evaluate(budget_allocation=p,
                                   n_simulations=n_simulations,
                                   parallel=parallel,
                                   num_cpu_cores=num_cpu_cores,
                                   statistic=statistic)
            list_prior_tf.append(m)
            list_prior_stderr.append(stderr)
        list_all_prior_tf.append(list_prior_tf)
        list_all_prior_stderr.append(list_prior_stderr)

        list_prior.append(prior)

        # save a description of what each strategy is
        with open(os.path.join(path_experiment, f'prior_parameter_{n_nodes}_nodes.txt'), 'w') as fi:
            fi.write(prior_parameter)

        # reduce the dimension of the input space by choosing to only allocate the budget between nodes with the highest
        # degrees. The function return the indices of these nodes
        # The indices correspond to the first item of the prior
        node_attributes = [degrees, capacities, transmissions]
        node_indices = bo_choose_sentinels(node_attributes=node_attributes,
                                           sentinels=sentinels,
                                           mode=mode_choose_sentinels)

        # compute the baseline, i.e., the expected value of the objective function for a uniform distribution of the
        # budget over all nodes (regardless of the number of sentinels)
        baseline_mean, baseline_stderr, x_baseline = bo_baseline(total_budget=total_budget,
                                                                 eval_function=f.evaluate,
                                                                 n_nodes=n_nodes,
                                                                 parallel=parallel,
                                                                 num_cpu_cores=num_cpu_cores,
                                                                 statistic=statistic)

        # ----------------------------------------
        # start the chosen optimizer
        # shared optimizer parameters
        save_test_strategies_path = os.path.join(path_sub_experiment, 'raw_data/test_strategies_history')
        if not os.path.exists(save_test_strategies_path) and save_test_strategies:
            os.makedirs(save_test_strategies_path)
        optimizer_kwargs = {'n_nodes': n_nodes, 'node_indices': node_indices, 'eval_function': f.evaluate,
                            'n_simulations': n_simulations, 'statistic': statistic, 'total_budget': total_budget,
                            'max_iterations': max_iterations, 'parallel': parallel,
                            'num_cpu_cores': num_cpu_cores, 'save_test_strategies': save_test_strategies,
                            'save_test_strategies_path': save_test_strategies_path}

        t0 = time()  # time for an individual optimization run
        if optimizer == 'cma':

            # CMA-ES can take only an initial population of one. For this the uniform baseline is used
            # TODO maybe change to/test with highest degree baseline ?
            optimizer_kwargs['initial_population'] = prior[cma_initial_population]
            optimizer_kwargs['bounds'] = bounds
            optimizer_kwargs['sigma'] = cma_sigma
            optimizer_kwargs['popsize'] = popsize

            # optimizers return the best test strategy, a history of the best solutions during a run (with stderror) and
            # the time it took to run the optimizer
            best_test_strategy, best_solution_history, stderr_history, \
            time_for_optimization = \
                bo_cma(**optimizer_kwargs)
            # TODO best_test_strategy needs to run through the softmax function, otherwise its wrong

        elif optimizer == 'gpgo':

            optimizer_kwargs['prior_x'] = prior
            optimizer_kwargs['prior_y'] = list_prior_tf
            optimizer_kwargs['prior_stderr'] = list_prior_stderr
            optimizer_kwargs['acquisition_function'] = acquisition_function
            optimizer_kwargs['use_prior'] = use_prior

            # optimizers return the best test strategy, a history of the best solutions during a run (with stderror) and
            # the time it took to run the optimizer
            best_test_strategy, best_solution_history, stderr_history, \
            time_for_optimization, time_acquisition_optimization, time_update_surrogate = \
                bo_pyGPGO(**optimizer_kwargs)

        elif optimizer == 'np':

            # # check validity of neural process hyperparameters
            if num_context + num_target > len(prior):
                raise ValueError('The context and target size together must not exceed the number of the budgets in the prior.'
                                 f'Got {num_context} + {num_target} > {len(prior)}')

            optimizer_kwargs['prior'] = prior
            optimizer_kwargs['prior_y'] = list_prior_tf
            optimizer_kwargs['prior_stderr'] = list_prior_stderr
            optimizer_kwargs['acquisition_function'] = acquisition_function
            optimizer_kwargs['epochs'] = epochs
            optimizer_kwargs['batch_size'] = batch_size
            optimizer_kwargs['r_dim'] = r_dim
            optimizer_kwargs['z_dim'] = z_dim
            optimizer_kwargs['h_dim'] = h_dim
            optimizer_kwargs['num_context'] = num_context
            optimizer_kwargs['num_target'] = num_target
            optimizer_kwargs['z_sample_size'] = z_sample_size

            # optimizers return the best test strategy, a history of the best solutions during a run (with stderror) and
            # the time it took to run the optimizer
            best_test_strategy, best_solution_history, stderr_history,\
            time_for_optimization, time_acquisition_optimization, time_update_surrogate = \
                bo_neural_process(**optimizer_kwargs)

        else:
            raise ValueError('Optimizer not supported')

        print('------------------------------------------------------')

        # plot and save to disk the results of the individual optimization runs
        bo_plot_optimizer_history(optimizer_history=best_solution_history, stderr_history=stderr_history,
                                  baseline_mean=baseline_mean, baseline_stderr=baseline_stderr,
                                  n_nodes=n_nodes, sentinels=sentinels,
                                  path_experiment=path_sub_experiment, optimizer=optimizer)

        bo_plot_time_for_optimization(time_for_optimization=time_for_optimization,
                                      path_experiment=path_sub_experiment,
                                      optimizer=optimizer)

        if optimizer == 'gpgo' or optimizer == 'np':
            bo_plot_time_for_optimization(time_for_optimization=time_acquisition_optimization,
                                          path_experiment=path_sub_experiment,
                                          optimizer=optimizer,
                                          file_name='time_for_acquisition_optimization.png',
                                          title='Time for acquisition function optimization')
            bo_plot_time_for_optimization(time_for_optimization=time_update_surrogate,
                                          path_experiment=path_sub_experiment,
                                          optimizer=optimizer,
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
        bo_save_results(best_test_strategy=best_test_strategy,
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
        if optimizer == 'gpgo' or optimizer == 'np':
            list_time_acquisition_optimization.append(time_acquisition_optimization)
            list_time_update_surrogate.append(time_update_surrogate)

    # save the results of the experiment
    raw_data_path = os.path.join(path_experiment, 'raw_data', str(n_runs_start))

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
    if optimizer == 'gpgo' or optimizer == 'np':
        kwargs_save_raw_data['list_time_acquisition_optimization'] = list_time_acquisition_optimization
        kwargs_save_raw_data['list_time_update_surrogate'] = list_time_update_surrogate
    bo_save_raw_data(**kwargs_save_raw_data)

    # create a .done file in the sub path to indicate the run is finished
    with open(os.path.join(raw_data_path, '.done'), 'w') as done_file:
        done_file.write(f'Individual run {n} finished')

    print(f'Optimization end: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')

    # ------------------------------------------------------------
    # postprocessing of all runs
    # ------------------------------------------------------------
    # compute the averages over all optimization runs of the prior, the optimizer, the baseline and their standard error

    # compute the average OTFs, baseline and their standard errors
    average_best_otf, average_best_otf_stderr = bo_compute_average_otf_and_stderr(list_otf=list_best_otf,
                                                                                  list_stderr=list_best_otf_stderr,
                                                                                  n_runs=n_runs)

    average_baseline, average_baseline_stderr = bo_compute_average_otf_and_stderr(list_otf=list_baseline_otf,
                                                                                  list_stderr=list_baseline_otf_stderr,
                                                                                  n_runs=n_runs)

    average_ratio_otf = np.mean(list_ratio_otf)

    output = f'Results averaged over {n_runs} optimizer runs' \
             f'\naverage ratio otf to baseline: {average_ratio_otf}' \
             f'\naverage baseline and stderr: {average_baseline}, {average_baseline_stderr}' \
             f'\naverage best strategy OTF and stderr: {average_best_otf}, {average_best_otf_stderr}' \
             f'\nTime for optimization (in hours): {(time() - time_start) / 3600}'

    bo_save_results(best_test_strategy=None,
                    save_test_strategy=False,
                    path_experiment=os.path.join(path_experiment, 'raw_data', str(n_runs_start)),
                    output=output)
    print(output)
