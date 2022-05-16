from email import policy
import numpy as np
from target_function_parameter import all_parameters_aggregation as tf_aggregation
from target_function_parameter import all_parameters_statistics as tf_statistics
from gym import ObservationWrapper, spaces
import gym
import gym
from stable_baselines3 import PPO
import pandas as pd
from pyoptes.optimization.budget_allocation import target_function as f
from scipy import special
import os
from CustomPolicy import ActorCriticPolicy

import gym
import torch as th
from stable_baselines3 import PPO

class CustomEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['console']}


  def __init__(self, n_nodes, network, net, id):
    super(CustomEnv, self).__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    self.nodes = n_nodes
    transmissions = pd.read_csv(f"/Users/admin/pyoptes/src/pyoptes/optimization/budget_allocation/supervised_learning/{network}/{n_nodes}/{net}{id}/dataset.txt", header = None)
    transmissions = transmissions[[2, 2, 0, 1, 3]]  
    capacities = pd.read_csv(f"/Users/admin/pyoptes/src/pyoptes/optimization/budget_allocation/supervised_learning/{network}/{n_nodes}/{net}{id}/barn_size.txt", header = None)
    capacities = capacities.iloc[0][:self.nodes].to_numpy()
    transmissions = transmissions.to_numpy()
    self.current_state = np.ones((n_nodes,))
    self.time_step = 0
    self.opt_budget = self.current_state
    self.best_reward = -np.inf

    # at the beginning, call prepare() once:
    f.prepare(
        use_real_data = False, #False = synthetic data
        static_network = None, #use waxman graph
        n_nodes = n_nodes, #size of network
        max_t = 365, #time horizon
        expected_time_of_first_infection = 30, #30 days
        capacity_distribution = capacities, #lambda size: np.ones(size), # any function accepting a 'size=' parameter
        delta_t_symptoms=60, #abort simulation after 60 days when symptoms first show up and testing becomes obsolet
        p_infection_by_transmission=0.5,
        pre_transmissions = transmissions #use pre-stored transmission data
        )
 
    self.evaluation_params = { 
        'aggregation' :  tf_aggregation,
        'statistic' : tf_statistics,
        'n_simulations' : 10000, 
        'parallel': True,
        'num_cpu_cores': -1
        }

    self.min_action = -100
    self.max_action = 100
    #action - budget shares vector  

    self.action_space = spaces.Box(
      low=self.min_action,
      high=self.max_action,
      shape=(self.nodes, ),
      dtype=np.float32
      )

    # obs = state : self.nodes probs infection, 4 metrics damage = output SI-model
    self.observation_space = spaces.Box(
      low = np.inf, 
      high = np.inf, 
      shape = (1, self.nodes), 
      dtype=np.float32) 


  def step(self, action):
    
    # Execute one time step within the environment
    # Each timestep, the agent chooses an action, and the environment returns an observation and a reward.        
    self.time_step+=1
    self.current_state = self.current_state + action

    #opt_budget  = self.current_state
    opt_budget = special.softmax(self.current_state, axis=0)*self.nodes

    ratio_infected, node_metrics, mean_sq_metrics, mean_metrics, p95_metrics = f.evaluate(opt_budget, **self.evaluation_params)
    
    #reward = - mean_sq_metrics[0]

    reward = - mean_sq_metrics[0]

    #if reward > self.best_reward:
    self.best_reward = reward
    print(opt_budget)
    #self.opt_budget = opt_budget

    print(f'time_step: {self.time_step}, best_reward: {self.best_reward}')

    done = bool(reward<-4000)

    info = {}

    return self.current_state, reward, done, info 

  def reset(self):
    # Reset the state of the environment to an initial state
    self.current_state = np.ones((self.nodes,))
    return self.current_state

models_dir = "/Users/admin/pyoptes/PPO/PPO_train"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = CustomEnv(n_nodes=120, network = "synthetic_networks", net = "syndata", id=0)


#define separate policies
policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=[dict(pi=[128, 128], vf=[128, 128])])

"""kwargs = dict(policy= "MlpPolicy", env= env, verbose= 1, tensorboard_log= logdir, policy_kwargs = policy_kwargs,
learning_rate = 0.0003, n_steps = 2048, batch_size = 64, n_epochs = 10, gamma = 0.99, gae_lambda = 0.95,
clip_range = 0.2, clip_range_vf = None, normalize_advantage = True, ent_coef = 0.0, vf_coef = 0.5,
max_grad_norm = 0.5, use_sde = False, sde_sample_freq = -1, target_kl = None, create_eval_env=False,
device='auto', _init_setup_model=True)"""

model = PPO(policy=ActorCriticPolicy, env=env, verbose=1, policy_kwargs=policy_kwargs, 
n_steps = 4, batch_size= 4, n_epochs = 10, target_kl = 0.03, learning_rate = 1)

obs = env.reset()

TIMESTEPS = 10000
for i in range(30):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/{TIMESTEPS*i}")