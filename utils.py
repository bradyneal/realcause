import numpy as np
import pandas as pd
import torch
from data import Z, T, Y
from types import FunctionType, MethodType


PANDAS = 'pandas'
TORCH = 'torch'


def to_data_format(data_format, z, t, y):
    format_to_func = {
        PANDAS: to_pandas_df,
        TORCH: to_tensors
    }
    if data_format in format_to_func.keys():
        return format_to_func[data_format](z, t, y)
    else:
        raise ValueError('Invalid data format: {} ... Valid formats: {}'.format(data_format, list(format_to_func.keys())))


def to_pandas_df(z, t, y):
    """
    Convert array-like z, t, and y to Pandas DataFrame
    :param z: 1d or 2d np array, list, or tuple of covariates
    :param t: 1d np array, list, or tuple of treatments
    :param y: 1d np array, list, or tuple of outcomes
    :return: Pandas DataFrame of z, t, and y
    """
    if isinstance(z, (list, tuple)):
        if any(isinstance(z_i, list, tuple) for z_i in z):
            d = {get_zlabel(i + 1): z_i for i, z_i in enumerate(z)}
        elif any(isinstance(z_i, np.ndarray) for z_i in z):
            assert all(z_i.ndim == 1 or z_i.shape[1] == 1 for z_i in z)
            d = {get_zlabel(i + 1): z_i for i, z_i in enumerate(z)}
        else:
            d = {Z: z}
    elif isinstance(z, np.ndarray):
        if z.ndim == 1:
            d = {Z: z}
        elif z.ndim == 2:
            # Assumes the examples are in the rows and covariates in the columns
            d = {get_zlabel(i + 1): z_i for i, z_i in enumerate(z.T)}
        else:
            raise ValueError('Unexpected z.ndim: {}'.format(z.ndim))
    else:
        print('Warning: unexpected z type: {}'.format(type(z)))
    d[T] = t
    d[Y] = y
    return pd.DataFrame(d)


def to_tensors(*args):
    if len(args) == 1:
        if isinstance(args[0], (tuple, list)):
            args = args[0]
        else:
            return torch.tensor(args[0], dtype=torch.float)
    return tuple(torch.tensor(arg, dtype=torch.float) for arg in args)


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
    if not isinstance(tensors, (list, tuple)):
        tensors = (tensors,)
    np_vects = tuple(to_np_vector(x, by_column=by_column, thin_interval=thin_interval) for x in tensors)
    if len(np_vects) == 1:
        return np_vects[0]
    else:
        return np_vects


def get_num_positional_args(f):
    if isinstance(f, FunctionType):
        n_args = f.__code__.co_argcount
    elif isinstance(f, MethodType):
        # Get corresponding function of class method and remove 'self' argument
        f = f.__func__
        n_args = f.__code__.co_argcount - 1
    else:
        raise ValueError('Invalid argument type: {}'.format(type(f)))

    if f.__defaults__ is not None:  # in case there are no kwargs
        n_kwargs = len(f.__defaults__)
    else:
        n_kwargs = 0

    n_positional_args = n_args - n_kwargs
    return n_positional_args


def get_zlabel(i=None, zlabel=Z):
    return zlabel if i is None else zlabel + str(i)
