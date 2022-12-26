'''
Collects and runs all different scripts used for the black-box optimization, generating results, plots and analysis.
'''

import argparse
from pyoptes import bbo_optimization, bbo_document_experiments
from pyoptes import bbo_combined_plots, bbo_create_individual_plots
from pyoptes import inspect_test_strategies, bbo_inspect_prior, bbo_sanity_check
from pyoptes import bbo_create_samples, bbo_explore_evaluation
from pyoptes import bbo_explore_target_function
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("mode",
                        choices=['optimization', 'document_experiments',
                                 'combined_plots', 'individual_plots',
                                 'inspect_test_strategies,', 'inspect_prior', 'sanity_check',
                                 'create_samples', 'explore_evaluation',
                                 'explore_target_function'],
                        default='optimization',
                        help="The mode of operation. Either 'inspect' or 'optimization'.")

    parser.add_argument("--sentinels", type=int, default=1040,
                        help="Set the number of nodes that are used. Has to be smaller than or equal to n_nodes. "
                             "Default is 120 nodes.")
    parser.add_argument("--n_nodes", type=int, default=1040, choices=[120, 1040, 57590],
                        help="Si-simulation parameter. "
                             "Defines the number of nodes used by the SI-model to create a graph. "
                             "Default value is 120 nodes.")

    # ------------------ optimization hyperparameters -------------------
    parser.add_argument("--optimizer", choices=['cma', 'gpgo', 'np'],
                        help="Choose the optimizer to run on the SI-model. Choose between CMA-ES, "
                             "Gaussian Process (GP) and Neural Process (NP).")
    parser.add_argument("--name_experiment",
                        help="The name of the folder where the results of the optimizer run are saved to.")
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

    parser.add_argument('--path_plot', default='../data/blackbox_learning/results/',
                        help="Optimizer parameter. Location where all the individual results"
                             " of the optimizers are saved to. "
                             "Default location is 'pyoptes/optimization/budget_allocation/blackbox_learning/plots/'")
    parser.add_argument('--path_networks', default='../../networks/data',
                        help='Location where the networks are saved to. '
                             'Path on cluster. /p/projects/ou/labs/gane/optes/mcmc_100nets/data'
                             '/p/projects/ou/labs/gane/optes/mcmc_100nets/data/')
    parser.add_argument('--path_data', default='../../data_pyoptes',
                        help="Specifies where the samples created by 'bbo_create_samples.py' are saved to. ")
    args = parser.parse_args()

    if args.mode == 'optimization':
        bbo_optimization(optimizer=args.optimizer,
                         name_experiment=args.name_experiment,
                         sentinels=args.sentinels,
                         n_nodes=args.n_nodes,
                         n_runs=args.n_runs,
                         n_runs_start=args.n_runs_start,
                         max_iterations=args.max_iterations,
                         acquisition_function=args.acquisition_function,
                         use_prior=args.use_prior,
                         prior_mixed_strategies=args.prior_mixed_strategies,
                         prior_only_baseline=args.prior_only_baseline,
                         r_dim=args.r_dim,
                         z_dim=args.z_dim,
                         h_dim=args.h_dim,
                         num_target=args.num_target,
                         num_context=args.num_context,
                         z_sample_size=args.z_sample_size,
                         epochs=args.epochs,
                         batch_size=args.batch_size,
                         popsize=args.popsize,
                         scale_sigma=args.scale_sigma,
                         cma_initial_population=args.cma_initial_population,
                         statistic_str=args.statistic,
                         n_simulations=args.n_simulations,
                         graph_type=args.graph_type,
                         scale_total_budget=args.scale_total_budget,
                         parallel=args.parallel,
                         num_cpu_cores=args.num_cpu_cores,
                         delta_t_symptoms=args.delta_t_symptoms,
                         p_infection_by_transmission=args.p_infection_by_transmission,
                         expected_time_of_first_infection=args.expected_time_of_first_infection,
                         mode_choose_sentinels=args.mode_choose_sentinels,
                         save_test_strategies=args.save_test_strategies,
                         plot_prior=args.plot_prior,
                         path_plot=args.path_plot,
                         path_networks=args.path_networks,
        )
    elif args.mode == 'inspect':
        # TODO fix faulty paths
        inspect_test_strategies(path_plot=args.path_plot)

    # TODO compute_baseline

    # TODO postprocessing

    elif args.mode == 'create_combined_plots':

        bbo_combined_plots(path_plot=args.path_plot,
                           optimizer=args.optimizer,
                            n_nodes=args.n_nodes,
                            sentinels=args.sentinels,
                            max_iterations=args.max_iterations,
                            acquisition_function=args.acquisition_function,
                            use_prior=args.use_prior,
                            prior_only_baseline=args.prior_only_baseline,
                            prior_mixed_strategies=args.prior_mixed_strategies,
                            popsize=args.popsize,
                            scale_sigma=args.scale_sigma,
                            statistic=args.statistic,
                            n_simulations=args.n_simulations,
                            graph_type=args.graph_type,
                            scale_total_budget=args.scale_total_budget,
                            mode_choose_sentinels=args.mode_choose_sentinels)

    elif args.mode == 'individual_plots':
        bbo_create_individual_plots(path_plot=args.path_plot)

    elif args.mode == 'document_experiments':
        bbo_document_experiments(path_plot=args.path_plot)

    elif args.mode == 'create_samples':
        bbo_create_samples(path_data=args.path_data,
                           n_nodes=args.n_nodes,
                           n_runs=args.n_runs,
                           n_simulations=args.n_simulations,
                           graph_type=args.graph_type,
                           scale_total_budget=args.scale_total_budget,
                           parallel=args.parallel,
                           num_cpu_cores=args.num_cpu_cores,
                           delta_t_symptoms=args.delta_t_symptoms,
                           p_infection_by_transmission=args.p_infection_by_transmission,
                           expected_time_of_first_infection=args.expected_time_of_first_infection,
                           sentinels=args.sentinels,
                           statistic_str=args.statistic,
                           path_networks=args.path_networks,)

    elif args.mode == 'explore_evaluation':
        bbo_explore_evaluation()

    elif args.mode == 'explore_evaluation':
        # call bbo_explore_target_function with the arguments from the command line
        bbo_explore_target_function(n_runs=args.n_runs,
                                    statistic_str=args.statistic,
                                    n_simulations=args.n_simulations,
                                    graph_type=args.graph_type,
                                    scale_total_budget=args.scale_total_budget,
                                    parallel=args.parallel,
                                    num_cpu_cores=args.num_cpu_cores,
                                    delta_t_symptoms=args.delta_t_symptoms,
                                    p_infection_by_transmission=args.p_infection_by_transmission,
                                    expected_time_of_first_infection=args.expected_time_of_first_infection,
                                    mode_choose_sentinels=args.mode_choose_sentinels,
                                    path_networks=args.path_networks)

    elif args.mode == 'inspect_prior':
        bbo_inspect_prior(path_plot=args.path_plot,)

    elif args.mode == 'sanity_check':

        # call bbo_sanity_check with the arguments from the command line
        bbo_sanity_check(n_nodes=args.n_nodes,
                         n_runs=args.n_runs,
                         graph_type=args.graph_type,
                         scale_total_budget=args.scale_total_budget,
                         path_networks=args.path_networks)

