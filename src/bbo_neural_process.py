'''
Use neural processes to approximate the SI-simulation
'''


from pyoptes.optimization.budget_allocation import target_function as f
from pyoptes import create_graph, rms_tia
from pyoptes import create_test_strategy_prior, map_low_dim_x_to_high_dim

import numpy as np
from tqdm import tqdm

import torch
from pyoptes import NeuralProcess, NeuralProcessTrainer, context_target_split, TrainingDataset
from torch.utils.data import DataLoader
from time import time

if __name__ == '__main__':

    print('Start')

    scale_total_budget = 1
    n_nodes = 120
    sentinels = 12
    graph_type = 'syn'
    n_runs = 2
    n_simulations = 2
    path_networks = '../../networks/data'
    statistic = rms_tia
    n = 0
    total_budget = scale_total_budget * n_nodes

    ###############################################################################

    transmissions, capacities, degrees = create_graph(n, graph_type, n_nodes, path_networks)

    f.prepare(n_nodes=n_nodes,
              capacity_distribution=capacities,
              pre_transmissions=transmissions,
              p_infection_by_transmission=0.5,
              delta_t_symptoms=60,
              expected_time_of_first_infection=30,
              static_network=None,
              use_real_data=False)

    # create a list of test strategies based on different heuristics
    prior, prior_node_indices, prior_parameter = \
        create_test_strategy_prior(n_nodes=n_nodes,
                                   node_degrees=degrees,
                                   node_capacities=capacities,
                                   total_budget=total_budget,
                                   sentinels=sentinels,
                                   mixed_strategies=True,
                                   only_baseline=False)

    # evaluate the strategies in the prior
    list_prior_tf = []
    list_prior_stderr = []

    for i, p in tqdm(enumerate(prior), leave=False, total=len(prior)):

        p = map_low_dim_x_to_high_dim(x=p,
                                      number_of_nodes=n_nodes,
                                      node_indices=prior_node_indices[i])

        m, stderr = f.evaluate(budget_allocation=p,
                               n_simulations=n_simulations,
                               parallel=True,
                               num_cpu_cores=-1,
                               statistic=statistic)
        list_prior_tf.append(m)
        list_prior_stderr.append(stderr)

    print('shape prior x and y, ', np.shape(prior), np.shape(list_prior_tf))

    ###############################################################################
    # training
    ###############################################################################
    # the neural process trainer expects data to be torch tensors and of type float
    # the data is expected in shape (batch_size, num_samples, function_dim)
    # (num_samples, function_dim) define how many different budgets are used

    x = torch.tensor(prior).unsqueeze(0).float()
    y = torch.tensor(list_prior_tf).unsqueeze(1).unsqueeze(0).float()
    print(x.size())
    print(y.size())

    # the dataset should consist of a list of (budget,y) pairs for each network
    # this means the neural process can be trained once, and used repeatedly in different experiments
    dataset = TrainingDataset(x, y)

    x_dim = 12
    y_dim = 1
    r_dim = 50  # Dimension of representation of context points
    z_dim = 50  # Dimension of sampled latent variable
    h_dim = 50  # Dimension of hidden layers in encoder and decoder

    neuralprocess = NeuralProcess(x_dim, y_dim, r_dim, z_dim, h_dim)

    batch_size = 10
    num_context = 3 # num_context + num_target has to be lower than num_samples
    num_target = 3

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(neuralprocess.parameters(), lr=3e-4)
    np_trainer = NeuralProcessTrainer(device, neuralprocess, optimizer,
                                      num_context_range=(num_context, num_context),
                                      num_extra_target_range=(num_target, num_target),
                                      print_freq=200)

    neuralprocess.training = True
    start = time()
    np_trainer.train(data_loader, 3000)
    print('Time for neural process training: ', time() - start)
    #################################################################################
    # predict SI-simulation output on a test budget
    #################################################################################
    target_budget = np.ones(12)*10
    # tensor needs shape (batch_size, num_samples, function_dim), function_dim is equal to the number of sentinels
    target_budget_tensor = torch.tensor(target_budget).float().unsqueeze(0).unsqueeze(0)
    print('shape target budget', target_budget_tensor.shape)

    neuralprocess.training = False

    for batch in data_loader:
        break
    x, y = batch
    x_context, y_context, _, _ = context_target_split(x[0:1], y[0:1],
                                                      num_context,
                                                      num_target)

    for _ in range(10):
        p_y_pred = neuralprocess(x_context, y_context, target_budget_tensor)
        # Extract mean of distribution
        mu = p_y_pred.loc.detach()
        sigma = p_y_pred.scale.detach()
        print('mu and sigma neural process', mu, sigma)

    print('target_budget', target_budget)
    p_mapped = map_low_dim_x_to_high_dim(x=target_budget,
                                         number_of_nodes=n_nodes,
                                         node_indices=prior_node_indices[0])

    m, stderr = f.evaluate(budget_allocation=p_mapped,
                           n_simulations=n_simulations,
                           parallel=True,
                           num_cpu_cores=-1,
                           statistic=statistic)

    print('\nmean and stderr', m, stderr, '\n')

    # --------------------------------------

    # use new measurement to update Neural Process
    neuralprocess.training = True
    # maybe create a new dataloader with only one entry

    print(x, x.size())
    print(target_budget_tensor, target_budget_tensor.size())

    x = torch.cat((x, target_budget_tensor), 1)
    mt = torch.tensor([m]).unsqueeze(0).unsqueeze(0).float()
    print('mt size', mt.size())
    y = torch.cat((y, mt), 1)
    print('new dataset shape', x.size(), y.size())
    print(x)
    print(y)
    new_dataset = TrainingDataset(x,y)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    np_trainer.train(data_loader, 3000)

    print('------------------')

    neuralprocess.training = False

    for batch in data_loader:
        break
    x, y = batch
    x_context, y_context, _, _ = context_target_split(x[0:1], y[0:1],
                                                      num_context,
                                                      num_target)

    for _ in range(10):
        p_y_pred = neuralprocess(x_context, y_context, target_budget_tensor)
        # Extract mean of distribution
        mu = p_y_pred.loc.detach()
        sigma = p_y_pred.scale.detach()
        print('mu and sigma neural process', mu, sigma)
    p_mapped = map_low_dim_x_to_high_dim(x=target_budget,
                                         number_of_nodes=n_nodes,
                                         node_indices=prior_node_indices[0])
    m, stderr = f.evaluate(budget_allocation=p_mapped,
                           n_simulations=n_simulations,
                           parallel=True,
                           num_cpu_cores=-1,
                           statistic=statistic)

    print('\nmean and stderr', m, stderr)