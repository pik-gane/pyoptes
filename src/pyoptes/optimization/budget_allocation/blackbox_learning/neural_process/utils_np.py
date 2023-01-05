import numpy as np
import torch
from torch.utils.data import Dataset


def bo_context_target_split(x, y, num_context, num_extra_target):
    """Given inputs x and their value y, return random subsets of points for
    context and target. Note that following conventions from "Empirical
    Evaluation of Neural Process Objectives" the context points are chosen as a
    subset of the target points.

    Parameters
    ----------
    x : torch.Tensor
        Shape (batch_size, num_points, x_dim)

    y : torch.Tensor
        Shape (batch_size, num_points, y_dim)

    num_context : int
        Number of context points.

    num_extra_target : int
        Number of additional target points.
    """
    # x is expected in shape (batch_size, num_samples, function_dim)
    # (2,100,1) // (2,100,120)
    # print('x.shape', np.shape(x))
    # print('x type', type(x))
    num_points = x.shape[1]
    # Sample locations of context and target points
    locations = np.random.choice(num_points,
                                 size=num_context + num_extra_target,
                                 replace=False)
    x_context = x[:, locations[:num_context], :]
    y_context = y[:, locations[:num_context], :]
    x_target = x[:, locations, :]
    y_target = y[:, locations, :]
    return x_context, y_context, x_target, y_target


class bo_TrainingDataset(Dataset):
    def __init__(self, x, y):
        self.data = [((x[i]), y[i])
                     for i in range(len(x))]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
