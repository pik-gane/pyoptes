import os
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("optimizer", choices=['cma', 'gpgo'],
                        help="Choose the optimizer to run on the SI-model. Choose between CMA-ES and GPGO")
    parser.add_argument("name_experiment",
                        help="The name of the folder where the results of the optimizer run are saved to.")

    # parser.add_argument("--sentinels", type=int, default=1040,
    #                     help="Set the number of nodes that are used. Has to be smaller than or equal to n_nodes. "
    #                          "Default is 120 nodes.")
    parser.add_argument("--n_nodes", type=int, default=1040, choices=[120, 1040, 57590],
                        help="Si-simulation parameter. "
                             "Defines the number of nodes used by the SI-model to create a graph. "
                             "Default value is 120 nodes.")
    # parser.add_argument('--n_runs', type=int, default=100,
    #                     help='The number of times the optimizer is run. Results are then averaged over all runs.'
    #                          'Default is 100 runs.')
    #
    # parser.add_argument("--max_iterations", type=int, default=50,
    #                     help="Optimizer parameter. The maximum number of iterations the algorithms run.")
    #
    # parser.add_argument('--acquisition_function', default='EI',
    #                     choices=['EI', 'PI', 'UCB', 'Entropy', 'tEI'],
    #                     help='GPGO optimizer parameter. Defines the acquisition function that is used by GPGO.')
    # parser.add_argument('--use_prior', type=bool, default=True,
    #                     help='GPGO optimizer parameter. Sets whether the surrogate function is fitted with priors '
    #                          'created by heuristics or by sampling random point. Only works when n_nodes and sentinels'
    #                          'are the same size. Default is True.')
    # parser.add_argument('--prior_mixed_strategies', type=bool, default=False,
    #                     help='GPGO optimizer parameter. '
    #                          'Sets whether to use test strategies that mix highest degrees and capacities in the prior.'
    #                          'If set to no the prior has the same shape for all network sizes.')
    # parser.add_argument('--prior_only_baseline', type=bool, default=False,
    #                     help='GPGO optimizer parameter. Sets whether to use only the baseline strategy in the prior.'
    #                          'If true the prior consists of only one item.')
    #
    # parser.add_argument('--popsize', type=int, default=9,
    #                     help='CMA-ES optimizer parameter. Defines the size of the population each iteration.'
    #                          'CMA default is "4+int(3*log(n_nodes))" '
    #                          '-> 18 of 120, 24 for 1040, 36 for 57590.'
    #                          'Is set to 9 for performance reasons.')
    # parser.add_argument('--scale_sigma', type=float, default=0.25,
    #                     help='CMA-ES optimizer parameter. Defines the scaling of the standard deviation. '
    #                          'Default is a standard deviation of 0.25 of the total budget.')
    # parser.add_argument('--cma_prior', type=int, default=0,
    #                     help='CMA-ES optimizer parameter. Sets which test strategy in the prior is used as the initial '
    #                          'population for cma.')
    #
    parser.add_argument("--statistic", choices=['mean', 'rms', '95perc'], default='rms',
                        help="Choose the statistic to be used by the target function. "
                             "Choose between mean, rms (root-mean-square) or 95perc (95th-percentile).")
    # parser.add_argument("--n_simulations", type=int, default=10000,
    #                     help="Si-simulation parameter. Sets the number of runs the for the SI-model. "
    #                          "Higher values of n_simulations lower the variance of the output of the simulation. "
    #                          "Default value is 1000.")
    # parser.add_argument('--graph_type', choices=['waxman', 'ba', 'syn'], default='syn',
    #                     help='Si-simulation parameter. Set the type of graph the simulation uses.'
    #                          ' Either Waxman,Synthetic or Barabasi-Albert (ba) can be used. Default is Synthetic.')
    parser.add_argument('--scale_total_budget', type=int, default=1, choices=[1, 4, 12],
                        help="SI-simulation parameter. Scales the total budget for SI-model. Default is 1.")
    #
    # parser.add_argument('--delta_t_symptoms', type=int, default=60,
    #                     help='Si-simulation parameter.. Sets the time (in days) after which an infection is detected'
    #                          ' automatically. Default is 60 days')
    # parser.add_argument('--p_infection_by_transmission', type=float, default=0.5,
    #                     help='Si-simulation parameter. The probability of how likely a trade animal '
    #                          'infects other animals. Default is 0.5.')
    # parser.add_argument('--expected_time_of_first_infection', type=int, default=30,
    #                     help='Si-simulation parameter. The expected time (in days) after which the first infection occurs. ')
    # parser.add_argument('--parallel', type=bool, default=True,
    #                     help='Si-simulation parameter. Sets whether multiple simulations run are to be done in parallel'
    #                          'or sequentially. Default is set to parallel computation.')
    parser.add_argument("--num_cpu_cores", type=int, default=32,
                        help='Si-simulation parameter. Defines the number of cpus to be used for the simulation '
                             'parallelization. If more cpus are chosen than available, the max available are selected.'
                             '-1 selects all available cpus. Default are 32 cpus.')
    #
    # parser.add_argument('--mode_choose_sentinels', choices=['degree', 'capacity', 'transmission'], default='degree',
    #                     help='Sets the mode of how sentinels are chosen. ')
    # parser.add_argument('--save_test_strategies', type=bool, default='',
    #                     help='Sets whether to save the test strategies that are evaluate in the optimization.')
    # parser.add_argument('--plot_prior', type=bool, default='',
    #                     help='')
    # parser.add_argument("--log_level", type=int, default=3, choices=range(1, 11), metavar="[1-10]",
    #                     help="Optimizer parameter. Only effects SMAC and GPGO. Sets how often log messages appear. "
    #                          "Lower values mean more messages.")
    # parser.add_argument('--path_plot', default='pyoptes/optimization/budget_allocation/blackbox_learning/plots/',
    #                     help="Optimizer parameter. Location where all the individual results"
    #                          " of the optimizers are saved to. "
    #                          "Default location is 'pyoptes/optimization/budget_allocation/blackbox_learning/plots/'")
    # parser.add_argument('--path_networks', default='../../networks/data',
    #                     help='Location where the networks are saved to. '
    #                          'Path on cluster. /p/projects/ou/labs/gane/optes/mcmc_100nets/data'
    #                          '/p/projects/ou/labs/gane/optes/mcmc_100nets/data/')
    #
    args = parser.parse_args()

    optimizer = args.optimizer
    name_experiment = args.name_experiment

    # sentinels = args.sentinels
    n_nodes = args.n_nodes
    # n_runs = args.n_runs

    # max_iterations = args.max_iterations

    # acquisition_function = args.acquisition_function
    # use_prior = args.use_prior
    # prior_mixed_strategies = args.prior_mixed_strategies
    # prior_only_baseline = args.prior_only_baseline

    # popsize = args.popsize
    # scale_sigma = args.scale_sigma
    # cma_prior = args.cma_prior

    statistic = args.statistic
    # n_simulations = args.n_simulations
    # graph_type = args.graph_type
    scale_total_budget = args.scale_total_budget

    # delta_t_symptoms = args.delta_t_symptoms
    # p_infection_by_transmission = args.p_infection_by_transmission
    # expected_time_of_first_infection = args.expected_time_of_first_infection
    # parallel = args.parallel
    num_cpu_cores = args.num_cpu_cores

    # mode_choose_sentinels = args.mode_choose_sentinels
    # save_test_strategies = args.save_test_strategies
    # plot_prior = args.plot_prior
    # log_level = args.log_level
    # path_plot = args.path_plot
    # path_networks = args.path_networks

    # map parameters to index for output-file
    o = {'n_nodes': {1040: 0, 57590: 1, 120: 2},
         'total_budget': {1: 0, 4: 1, 12: 2},
         'statistic': {'rms': 0, 'mean': 1, '95perc': 2}}

    output_file_name = f'{args.optimizer}_' \
                       f'{o["n_nodes"][args.n_nodes]}_' \
                       f'{o["statistic"][args.statistic]}_' \
                       f'{o["total_budget"][args.scale_total_budget]}'

    sbatch_parameters = '#!/bin/bash\n\n' \
                        '#SBATCH --constraint=broadwell\n\n' \
                        '#SBATCH --qos=short\n\n' \
                        f'#SBATCH --job-name=loebkens_{output_file_name}\n\n' \
                        '#SBATCH --account=gane\n\n' \
                        f'#SBATCH --output=logs/outputs_{output_file_name}.out\n\n' \
                        f'#SBATCH --error=logs/errors_{output_file_name}.err\n\n' \
                        '#SBATCH --workdir=/home/loebkens\n\n' \
                        '#SBATCH --nodes=1      # nodes requested\n\n' \
                        '#SBATCH --ntasks=1      # tasks requested\n\n' \
                        f'#SBATCH --cpus-per-task={args.num_cpu_cores}\n\n' \
                        '#SBATCH --mem=64000\n\n' \
                        'module load anaconda/5.0.0_py3\n' \
                        'source activate bbo\n' \

    bb_optimization = 'srun -n $SLURM_NTASKS python3 /home/loebkens/pyoptes/src/bb_optimization.py' \
                      f' {args.optimizer}' \
                      f' {args.name_experiment}' \

    optional_params = ' --path_plot /home/loebkens/pyoptes/src/pyoptes/optimization/budget_allocation/blackbox_learning/plots/' \
                      ' --path_networks /p/projects/ou/labs/gane/optes/mcmc_100nets/data/' \
                      " --graph syn" \
                      " --prior_mixed_strategies ''" \
                      ' --n_nodes 1040' \
                      " --sentinels 1040" \
                      " --statistic rms" \
                      " --scale_total_budget 1"

    script_content = sbatch_parameters + bb_optimization + optional_params

    with open('cluster scripts/test.sh', 'w') as f:
        f.write(script_content)
