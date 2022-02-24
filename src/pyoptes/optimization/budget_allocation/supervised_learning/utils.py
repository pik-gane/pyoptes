from ast import Str
from pickle import FALSE
from xmlrpc.client import boolean
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pyoptes.optimization.budget_allocation.supervised_learning import NN as nets
import torch
import torchvision
from sklearn.metrics import explained_variance_score, mean_squared_error
from torch.autograd import grad
from torch import optim

class device():
  def get_device():
      if torch.cuda.is_available():
          device = 'cuda:0'
      else:
          device = 'cpu'
      return device


class Loader(Dataset):
    def __init__(self, input_path, targets_path, path):

        if path == True:
          self.inputs = pd.read_csv(input_path, header = None, sep = ',')
          self.targets = pd.read_csv(targets_path, header = None, sep = ',')
        else:
          self.inputs = input_path 
          self.targets = targets_path
          
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        inputs = self.inputs.iloc[idx]
        targets = np.sqrt(self.targets.iloc[idx])
        return np.array(inputs), np.array(targets)


class processing():
  def postprocessing(train_input, train_targets, split, grads):

      train_input_data = pd.read_csv(train_input, header = None, sep = ',')

      train_targets_data = pd.read_csv(train_targets, header = None, sep = ',')
      
      is_NaN = train_input_data.isnull()
      row_has_NaN = is_NaN.any(axis=1)
      rows_with_NaN = train_input_data[row_has_NaN]
      del_cells = rows_with_NaN.index.values

      train_input_data = train_input_data.drop(del_cells)
      train_targets_data = train_targets_data.drop(del_cells)

      is_NaN = train_input_data.isnull()
      row_has_NaN = is_NaN.any(axis=1)

      #print(row_has_NaN)

      subset_training_input = train_input_data.iloc[split:]

      subset_training_targets = train_targets_data.iloc[split:]

      subset_val_input = train_input_data.iloc[0: split]
      subset_val_targets = train_targets_data.iloc[0: split]
      
      print(f'training size: {len(subset_training_input)} {len(subset_training_targets)}, test size: {len(subset_val_input)} {len(subset_val_targets)}')

      if grads == False:
        training_set = Loader(input_path = subset_training_input, 
                        targets_path = subset_training_targets, path = False)

        validation_set = Loader(input_path = subset_val_input, 
                          targets_path = subset_val_targets, path = False)

        return training_set, validation_set
      
      if grads == True:
        return subset_val_input, subset_val_targets
        
class model_selection():
  def set_model(model: str, dim: int, hidden_dims):
      if model == "Linear":
          model = nets.LinearNetwork(dim, 1, bias = True, hidden_dims = hidden_dims)
      elif model == "RNN":
          model = nets.RNNetwork(dim, 1, bias = True, hidden_dims = hidden_dims)
      elif model == "FCN":
          model = nets.FCNetwork(dim, 1, bias = True, hidden_dims = hidden_dims)
      #elif model == "GRU":
      #    model = nets.GRU(dim, 1, bias = True)
      return model


class training_process():
  def train(trainloader: DataLoader, model: torchvision.models, device: torch.device, 
            optimizer: torch.optim, criterion: torch.nn , verbose: int = 20):

      model.train()
      true = []
      pred = []
      train_loss = []

      for i, (inputs, targets) in enumerate(trainloader, 1):
          inputs, targets = inputs.to(device).float(), targets.to(device).float()

          optimizer.zero_grad()
          output = model.forward(inputs)
          #print(output[0], targets[0])
          loss = criterion(output, targets)
          loss.backward()
          optimizer.step()
          train_loss.append(loss.item())

          for j, val in enumerate(output):
              true.append(targets[j].item())
              pred.append(output[j].item())

      acc = explained_variance_score(true, pred)
      return np.mean(train_loss), acc

  def validate(valloader: DataLoader, model: torchvision.models,device: torch.device, 
              criterion: torch.nn, verbose: int = 20):
    
        model.eval()
        true = []
        pred = []
        val_loss = []
        with torch.no_grad():

            for i, (inputs, targets) in enumerate(valloader, 1):
                inputs, targets = inputs.to(device).float(), targets.to(device).float()
                output = model.forward(inputs)
                loss = criterion(output, targets)
                
                val_loss.append(loss.item())
                for j, val in enumerate(output):
                    true.append(targets[j].item())
                    pred.append(output[j].item())
                    
        acc = explained_variance_score(true, pred) #1 - var(y-y_hat)/var(y) 
        return np.mean(val_loss), acc
      
  def optimise(input_budget, targets, model, optimiser, device: torch.device, criterion: torch.nn , verbose: int = 20):
      model.train()
      
      input_budget = torch.reshape(input_budget, (1,input_budget.shape[0]))
      targets = torch.reshape(targets, (1,1))
      
      def softmax(x):
          out = np.exp(x)/sum(np.exp(x))
          return out

      inputs, targets = input_budget.to(device).float(), targets.to(device).float()
      
      optimiser.zero_grad()
      
      output = model.forward(inputs)

      loss = criterion(output, targets)

      loss.backward(retain_graph=True) 

      optimiser.step()

      #d_loss_dx = grad(outputs=loss, inputs=inputs, create_graph=False)
      #print(f'dloss/dx:\n {d_loss_dx}')
      #with torch.no_grad():
      #      for param in optimiser.param_groups[0]["params"]:
      #          param = softmax(param)
                #print(param)

      grads = optimiser.param_groups[0]["params"][0].detach()

      #grads.append(optimiser.param_groups)

      #acc = explained_variance_score(output.detach(), targets.detach()) 

      return loss, softmax(grads)*120#acc
  

  def evaluate(budget: torch.tensor, model: torchvision.models, device: torch.device):
    model.eval()
    input_budget = torch.reshape(budget, (1,budget.shape[0])) #clamp(0,121)
    budget = input_budget.to(device).float()
    output = model.forward(budget).to(device)
    return output.item()
