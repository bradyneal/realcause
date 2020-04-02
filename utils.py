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


def get_num_positional_args(f):
    n_args = f.__code__.co_argcount
    if f.__defaults__ is not None:  # in case there are no kwargs
        n_kwargs = len(f.__defaults__)
    else:
        n_kwargs = 0
    n_positional_args = n_args - n_kwargs
    return n_positional_args