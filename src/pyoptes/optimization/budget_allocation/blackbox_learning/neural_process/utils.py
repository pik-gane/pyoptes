import numpy as np
import torch


def context_target_split(x, y, num_context, num_extra_target):
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


def random_context_target_mask(img_size, num_context, num_extra_target):
    """Returns random context and target masks where 0 corresponds to a hidden
    value and 1 to a visible value. The visible pixels in the context mask are
    a subset of the ones in the target mask.

    Parameters
    ----------
    img_size : tuple of ints
        E.g. (1, 32, 32) for grayscale or (3, 64, 64) for RGB.

    num_context : int
        Number of context points.

    num_extra_target : int
        Number of additional target points.
    """
    _, height, width = img_size
    # Sample integers without replacement between 0 and the total number of
    # pixels. The measurements array will then contain pixel indices
    # corresponding to locations where pixels will be visible.
    measurements = np.random.choice(range(height * width),
                                    size=num_context + num_extra_target,
                                    replace=False)
    # Create empty masks
    context_mask = torch.zeros(width, height).byte()
    target_mask = torch.zeros(width, height).byte()
    # Update mask with measurements
    for i, m in enumerate(measurements):
        row = int(m / width)
        col = m % width
        target_mask[row, col] = 1
        if i < num_context:
            context_mask[row, col] = 1
    return context_mask, target_mask


def batch_context_target_mask(img_size, num_context, num_extra_target,
                              batch_size, repeat=False):
    """Returns bacth of context and target masks, where the visible pixels in
    the context mask are a subset of those in the target mask.

    Parameters
    ----------
    img_size : see random_context_target_mask

    num_context : see random_context_target_mask

    num_extra_target : see random_context_target_mask

    batch_size : int
        Number of masks to create.

    repeat : bool
        If True, repeats one mask across batch.
    """
    context_mask_batch = torch.zeros(batch_size, *img_size[1:]).byte()
    target_mask_batch = torch.zeros(batch_size, *img_size[1:]).byte()
    if repeat:
        context_mask, target_mask = random_context_target_mask(img_size,
                                                               num_context,
                                                               num_extra_target)
        for i in range(batch_size):
            context_mask_batch[i] = context_mask
            target_mask_batch[i] = target_mask
    else:
        for i in range(batch_size):
            context_mask, target_mask = random_context_target_mask(img_size,
                                                                   num_context,
                                                                   num_extra_target)
            context_mask_batch[i] = context_mask
            target_mask_batch[i] = target_mask
    return context_mask_batch, target_mask_batch
