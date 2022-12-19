import numpy as np
from torch.utils.data import Dataset


class GNNTrainingDataset(Dataset):
    def __init__(self, x, y):
        self.data = [((x[i]), y[i])
                     for i in range(len(x))]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":

    pass
