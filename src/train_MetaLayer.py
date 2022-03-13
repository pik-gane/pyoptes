from torch import nn
import torch
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from pyoptes import set_seed
import torch
from pyoptes.optimization.budget_allocation.supervised_learning.utils import Loader as Loader, device
from pyoptes.optimization.budget_allocation.supervised_learning.utils import processing as process
from pyoptes.optimization.budget_allocation.supervised_learning.utils import model_selection as model_selection
from torch_geometric.loader import DataLoader
from torch import optim
from sklearn.metrics import explained_variance_score
from Graph_DataList import prepare_convolutions as prep_conv
import torch.optim as optim
import torchvision
from torch.utils.tensorboard import SummaryWriter
from Meta_Layer import meta_layer as meta_layer




def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device

device = get_device()
set_seed(1)

"""implement tensorboard"""
#writer = SummaryWriter(log_dir = "/Users/admin/pyoptes_graphs/metalayer")

#load our data
inputs = "/Users/admin/pyoptes/src/pyoptes/optimization/budget_allocation/supervised_learning/training_data/wx_inputs.csv"
targets = "/Users/admin/pyoptes/src/pyoptes/optimization/budget_allocation/supervised_learning/training_data/wx_targets.csv"

#inputs = "/Users/admin/pyoptes/src/pyoptes/optimization/budget_allocation/supervised_learning/training_data/wx_inputs.csv"
#targets = "/Users/admin/pyoptes/src/pyoptes/optimization/budget_allocation/supervised_learning/training_data/wx_targets.csv"


#warp data into a loader
x, y = process.postprocessing(inputs, targets, split = 20000, grads = True)

#create a data_list consisting of a number of single identical graphs with different budget allocations
data_list = prep_conv(x,y)

#split data into training and test data
train_loader = DataLoader(data_list[5000:], batch_size = 128, shuffle = True)
test_loader = DataLoader(data_list[:5000], batch_size = 128, shuffle = True)

#define our graph neural network 
model = meta_layer(ins_nodes = 2, ins_edges = 1, ins_graphs = 6, hiddens= 16, outs = 6).double()
model.to(device)
epochs = 50

#define loss criterion
criterion = nn.L1Loss() #mean absolute error np.abs(y-y_hat)

lr = -2.5
optimizer_params = {"lr": 10**lr, "weight_decay": 0.005, "betas": (0.9, 0.999)}
optimizer = optim.AdamW(model.parameters(), **optimizer_params)

"""(optional) load pre-trained weights"""
#model.load_state_dict(torch.load("/Users/admin/pyoptes/src/pyoptes/optimization/budget_allocation/supervised_learning/trained_nn_states/meta_layer.pth"))

def training(loader, model, criterion, optimizer):
    model.train()
    true = []
    pred = []
    train_loss = []
    acc = []


    for batch in loader:
        optimizer.zero_grad()
        targets = batch.y.unsqueeze(-1) #= [32,1]
        targets = targets.to(device)

        x, edge_index, edge_weight, u, batch = batch.x, batch.edge_index, batch.weight, batch.edge_attr, batch.batch
        x, edge_index, edge_weight, u, batch = x.to(device), edge_index.to(device), edge_weight.to(device), u.to(device), batch.to(device)
        x, edge_weight, u = model.forward(x = x, edge_attr = edge_weight, u = u, edge_index = edge_index, batch = batch)
        
        #print(u.shape)
        #print(x.shape, targets.shape)
        loss = criterion(x, targets)
        loss.backward()
        optimizer.step()

        #print(u[0].item(), targets[0].item())

        train_loss.append(loss.item())

        #for j, val in enumerate(u):
            #true.append(targets[j].item())
            #pred.append(u[j].item())
        acc.append(explained_variance_score(targets.detach(), x.detach()))

    #acc = explained_variance_score(true, pred)
    return np.mean(train_loss), np.mean(acc)


def validate(valloader: DataLoader, model: torchvision.models):    
    model.eval()
    true = []
    pred = []
    val_loss = []
    acc = []

    with torch.no_grad():

        for batch in valloader:
            
            targets = batch.y.unsqueeze(-1) #= [32,1]
            targets = targets.to(device)

            x, edge_index, edge_weight, u, batch = batch.x, batch.edge_index, batch.weight, batch.edge_attr, batch.batch
            x, edge_index, edge_weight, u, batch = x.to(device), edge_index.to(device), edge_weight.to(device), u.to(device), batch.to(device)
            x, edge_weight, u = model.forward(x = x, edge_attr = edge_weight, u = u, edge_index = edge_index, batch = batch)
            
            loss = criterion(x, targets)

            val_loss.append(loss.item())

            #for j, val in enumerate(u):
                #true.append(targets[j].item())
                #pred.append(u[j].item())
            acc.append(explained_variance_score(targets.detach(), x.detach()))

    #acc = explained_variance_score(true, pred)
    return np.mean(val_loss), np.mean(acc)



total_loss = []
total_acc = []

_val_loss = []
_val_acc = []

train_loss_prev = np.inf
val_loss_prev = np.inf

for epoch in range(epochs):

  train_loss, train_acc = training(train_loader, model, criterion, optimizer) 
  total_loss.append(train_loss)
  total_acc.append(train_acc)
  
  val_loss, val_acc = validate(test_loader, model) 
  _val_loss.append(val_loss)
  _val_acc.append(val_acc)

  #writer.add_scalar(f'Loss/test nodes', val_loss, epoch)
  #writer.add_scalar(f'Accuracy/test nodes', val_acc, epoch)

  if train_loss < train_loss_prev:
    train_loss_prev = train_loss
    val_loss_prev = val_loss
    torch.save(model.state_dict(), "/Users/admin/pyoptes/src/pyoptes/optimization/budget_allocation/supervised_learning/trained_nn_states/meta_layer.pth")

    print(f'epoch: {epoch+1}, train loss: {train_loss_prev}, train acc: {train_acc}, val loss: {val_loss_prev}, val acc: {val_acc}')

plt.figure()
plt.plot(np.arange(epochs), total_loss, label = "training loss")
plt.plot(np.arange(epochs), _val_loss, label = "val loss")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend()

plt.figure()
plt.plot(np.arange(epochs), total_acc, label = "training acc")
plt.plot(np.arange(epochs), _val_acc, label = "val acc")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()

plt.show()