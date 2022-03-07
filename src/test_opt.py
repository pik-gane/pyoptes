"""
import os
from tkinter import W
from xml.dom import HierarchyRequestErr
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from sklearn.metrics import explained_variance_score, mean_squared_error
import numpy as np
import torchvision
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import random
from tqdm import tqdm
from operator import xor
from scipy.stats import gaussian_kde as kde
import pylab as plt
from pyoptes import set_seed
from pyoptes.optimization.budget_allocation import target_function as f
import csv
import warnings
import networkx as nx
from pyoptes.optimization.budget_allocation.supervised_learning.utils import Loader as Loader
from pyoptes.optimization.budget_allocation.supervised_learning.utils import processing as process
from pyoptes.optimization.budget_allocation.supervised_learning.utils import model_selection as model_selection
from pyoptes.optimization.budget_allocation.supervised_learning.utils import device as get_device
from pyoptes.optimization.budget_allocation.supervised_learning.utils import training_process as train_nn
import torch
from pyoptes.optimization.budget_allocation import target_function as f

from pyoptes import set_seed

set_seed(1)

print("Preparing the target function for a random but fixed transmissions network")
# generate a Waxman graph:
waxman = nx.waxman_graph(120)
pos = dict(waxman.nodes.data('pos'))
# convert into a directed graph:
static_network = nx.DiGraph(nx.to_numpy_array(waxman))

# at the beginning, call prepare() once:
f.prepare(
  use_real_data=False, 
  static_network=static_network,
  n_nodes=120,
  max_t=365, 
  expected_time_of_first_infection=30, 
  capacity_distribution = np.random.lognormal, #lambda size: np.ones(size), # any function accepting a 'size=' parameter
  delta_t_symptoms=60
  )

n_trials = 100

n_inputs = f.get_n_inputs()
total_budget = n_inputs

evaluation_parms = { 
        'n_simulations': 100, 
        'statistic': lambda a: (np.mean(a**2), np.std(a**2)/np.sqrt(a.size)) #lambda a: np.percentile(a, 95)
        }

x4 = [2.8914362880045017,0.35831236525409343,0.2296649261898272,0.3425690550365125,1.7109489009800984,0.3443093783270971,3.963609151596222,0.2410259867106228,0.5305140830693033,0.18340502014290386,0.5055061405768808,0.26676783581066277,0.1654998589067721,0.2318213909468798,0.2802419775234741,0.5733646532732881,0.1508804848292106,0.33925187545398394,0.5894423486892597,0.35585043490043006,1.5616796284204166,0.21650593945105895,0.2454788181592601,1.3863678746385075,0.5526862508749875,0.3057229266107217,0.6264167076029503,1.0599473062071236,3.224943629549123,38.26714660982909,0.20884442969281183,0.17567782863157969,0.18681805997560588,0.1706237143586993,3.034232386129107,0.17705006279444757,0.6796271013012947,0.40895203370699634,0.2739560396739501,0.20572883382505033,0.2413538872419512,0.19874706201139214,1.1742269007690402,0.45478685581304223,0.17541281119847918,0.2605315414331145,0.19725789278894743,0.14652513877633264,0.9595641180768661,0.5598448761160758,0.2276530314581291,0.5847029594385394,0.21924742305334075,0.20663420124633758,1.3755026024682175,3.597490484595193,0.28731630891097276,2.010163248112303,0.9741356389506444,0.2873482807600725,0.22744261225226198,0.4278490598229752,0.24684246975004714,0.35723547128772615,7.488937056551736,0.5207915044646086,0.2239297549643748,0.1748533260562903,0.1802883108886866,0.29557501142082954,0.24319919893738473,0.6386812612153876,1.4890501529726097,0.18172708379042862,0.8115433310117546,2.1474976940122192,2.3233839375189493,0.17154993907552138,0.15905047793509125,0.14787763379829383,0.21983788268614243,0.16499268519897353,0.20230792441745482,0.21907557329844385,2.0920893199065174,0.2577332376825907,0.19482966196202775,0.28064284278179713,0.343149486144411,0.16755196997006616,0.5004851277139263,0.7736839939815312,0.7344297743283253,0.967706453536394,0.22130017293258086,0.2600319159031693,0.1630216707171861,0.1927344432425598,3.273717678179282,0.21171499294760918,1.110069723921749,0.9396603955962821,1.3277227016346134,0.1792091340973688,0.8849623643414923,0.3451501051732689,0.18184323583911496,0.2129493603056851,0.18477905094472385,0.24548540002200742,0.17148208347663652,0.17601082027378304,0.18282542847086436,0.4449748988478477,0.3024858596314913,0.8634021289354473,0.19918422281908296,0.35526518419657616,0.14902084248590503,0.28390136285404066]

f_eval = []
si_err = []

for n in tqdm(range(n_trials)):
  tf, si = f.evaluate(x4, **evaluation_parms)
  print(np.sqrt(f_eval), si_err/(2*np.sqrt(f_eval)))
  f_eval.append(tf)
  si_err.append(si)
si_out,  si_out_err = np.mean(), np.mean(si_err/(2*np.sqrt(f_eval))) 
print(f'number infected animals: {si_out}, std err: {si_out_err}')

"""


import os
from tkinter import W
from xml.dom import HierarchyRequestErr
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from sklearn.metrics import explained_variance_score, mean_squared_error
import numpy as np
import torchvision
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import random
from tqdm import tqdm
from operator import xor
from scipy.stats import gaussian_kde as kde
import pylab as plt
from pyoptes import set_seed
from pyoptes.optimization.budget_allocation import target_function as f
import csv
import warnings
import networkx as nx
from pyoptes.optimization.budget_allocation.supervised_learning.utils import Loader as Loader
from pyoptes.optimization.budget_allocation.supervised_learning.utils import processing as process
from pyoptes.optimization.budget_allocation.supervised_learning.utils import model_selection as model_selection
from pyoptes.optimization.budget_allocation.supervised_learning.utils import device as get_device
from pyoptes.optimization.budget_allocation.supervised_learning.utils import training_process as train_nn
import torch
from pyoptes.optimization.budget_allocation import target_function as f

from pyoptes import set_seed

set_seed(1)

print("Preparing the target function for a random but fixed transmissions network")
# generate a Waxman graph:
waxman = nx.waxman_graph(120)
pos = dict(waxman.nodes.data('pos'))
# convert into a directed graph:
static_network = nx.DiGraph(nx.to_numpy_array(waxman))

# at the beginning, call prepare() once:
f.prepare(
  use_real_data=False, 
  static_network=static_network,
  n_nodes=120,
  max_t=365, 
  expected_time_of_first_infection=30, 
  capacity_distribution = np.random.lognormal, #lambda size: np.ones(size), # any function accepting a 'size=' parameter
  delta_t_symptoms=60
  )

n_inputs = f.get_n_inputs()
total_budget = n_inputs

evaluation_parms = { 
        'n_simulations': 100000, 
        'statistic': lambda a: (np.mean(a**2), np.std(a**2)/np.sqrt(a.size)) #lambda a: np.percentile(a, 95)
        }

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hidden_dims = (128, 64, 32, 16)
nodes = 120
pick = "RNN"

model = model_selection.set_model(pick, dim = nodes, hidden_dims = hidden_dims)
model.to(device)

#model.load_state_dict(torch.load("/Users/admin/pyoptes/src/ba_120_rnn.pth"))
model.load_state_dict(torch.load("/Users/admin/pyoptes/src/ba_120_rnn.pth"))

#for param_tensor in model.state_dict():
#    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

criterion = nn.L1Loss() #mean absolut error

inputs = "/Users/admin/pyoptes/src/inputs_ba_120_final.csv"
targets = "/Users/admin/pyoptes/src/targets_ba_120_final.csv"

model.requires_grad_(False)
train, test = process.postprocessing(inputs, targets, split = 10000, grads = False)

train_data = DataLoader(train, batch_size = 128, shuffle=True)
test_data = DataLoader(test, batch_size = 128, shuffle=True)

test_x, test_y = process.postprocessing(inputs, targets, split = 10000, grads = True)

test_x = test_x.to_numpy()
test_y = test_y.to_numpy()
initial_budget = test_x[10] #?makes a difference wether I use a.e. Sentinel based BudDist as Init or a a.e. random BD

test_x = torch.tensor(initial_budget).requires_grad_(True)
test_y = torch.tensor(np.zeros_like(test_y[0]))

val_loss, val_acc = train_nn.validate(valloader= test_data, model=model, device=device, criterion=criterion, verbose=10)
print(f'\n\nloss of model: {val_loss}, accuray of model: {val_acc}\n\n')

degree_values = sorted(static_network.degree, key=lambda x: x[1], reverse=True)

hd = []
for i in range(10):
  hd.append(degree_values[i][0])

print(f'nodes with highest degree: {hd}\n')

sentinels = hd
weights = np.zeros(n_inputs)
weights[sentinels] = 1
shares = weights / weights.sum()

x4 = shares * total_budget

test_x = torch.tensor(initial_budget).requires_grad_(True)
test_y = torch.tensor(0)

def evaluate(inputs, model, device):

    model.eval()

    inputs = inputs.to(device).float()

    inputs = inputs.unsqueeze(0)

    output = model.forward(inputs)

    return output.item()


f_eval, si_out_sq_err = f.evaluate(initial_budget, **evaluation_parms)
si_out_base,  si_out_err = np.sqrt(f_eval), si_out_sq_err/(2*np.sqrt(f_eval)) #

nn_out = evaluate(test_x, model, device = device)
nn_out = nn_out**2
print(f'\nloss of model: {si_out_base, nn_out}')