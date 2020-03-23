import numpy as np
import pandas as pd
import torch


def to_np_vector(x, by_column=True, thin_interval=None):
    if not isinstance(x, torch.Tensor):
        raise ValueError('Invalid input type: {}'.format(type(x)))
    if by_column:
        order = 'F'
    else:
        order = 'C' # by row
    np_vect = x.detach().numpy().reshape(-1, order=order)
    if thin_interval is not None:
        return np_vect[::thin_interval]
    else:
        return np_vect


def to_np_vectors(tensors, by_column=True, thin_interval=None):
    if not (isinstance(tensors, list) or isinstance(tensors, tuple)):
        tensors = (tensors,)
    np_vects = tuple(to_np_vector(x) for x in tensors)
    if len(np_vects) == 1:
        return np_vects[0]
    else:
        return np_vects
