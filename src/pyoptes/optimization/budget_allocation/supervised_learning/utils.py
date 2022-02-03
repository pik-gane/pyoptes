import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

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
        targets = self.targets.iloc[idx]
        return np.array(inputs), np.array(targets)
