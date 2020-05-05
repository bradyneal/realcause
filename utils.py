import numpy as np
import pandas as pd
import torch
from types import FunctionType, MethodType
import warnings

W = 'w'
T = 't'
Y = 'y'

PANDAS = 'pandas'
TORCH = 'torch'
NUMPY = 'numpy'


def to_data_format(data_format, w, t, y):
    format_to_func = {
        PANDAS: to_pandas_df,
        TORCH: to_tensors,
        NUMPY: to_np_arrays
    }
    if data_format in format_to_func.keys():
        return format_to_func[data_format](w, t, y)
    else:
        raise ValueError('Invalid data format: {} ... Valid formats: {}'.format(data_format, list(format_to_func.keys())))


def to_pandas_df(w, t, y):
    """
    Convert array-like w, t, and y to Pandas DataFrame
    :param w: 1d or 2d np array, list, or tuple of covariates
    :param t: 1d np array, list, or tuple of treatments
    :param y: 1d np array, list, or tuple of outcomes
    :return: Pandas DataFrame of w, t, and y
    """
    if isinstance(w, (list, tuple)):
        if any(isinstance(w_i, list, tuple) for w_i in w):
            d = {get_wlabel(i + 1): w_i for i, w_i in enumerate(w)}
        elif any(isinstance(w_i, np.ndarray) for w_i in w):
            assert all(w_i.ndim == 1 or w_i.shape[1] == 1 for w_i in w)
            d = {get_wlabel(i + 1): w_i for i, w_i in enumerate(w)}
        else:
            d = {W: w}
    elif isinstance(w, np.ndarray):
        if w.ndim == 1:
            d = {W: w}
        elif w.ndim == 2:
            # Assumes the examples are in the rows and covariates in the columns
            d = {get_wlabel(i + 1): w_i for i, w_i in enumerate(w.T)}
        else:
            raise ValueError('Unexpected w.ndim: {}'.format(w.ndim))
    else:
        warnings.warn(' unexpected w type: {}'.format(type(w)), Warning)
    d[T] = t
    d[Y] = y
    return pd.DataFrame(d)


def to_tensor(x):
    if isinstance(x, (pd.DataFrame, pd.Series)):
        x = x.values
    return torch.tensor(x, dtype=torch.float)


def to_tensors(*args):
    if len(args) == 1:
        if isinstance(args[0], (tuple, list)):
            args = args[0]
        else:
            return to_tensor(args[0])
    return tuple(to_tensor(arg) for arg in args)


def to_np_arrays(*args):
    if len(args) == 1:
        if isinstance(args[0], (tuple, list)):
            args = args[0]
        else:
            return np.array(args[0], dtype=np.float)
        return np.array(args[0], dtype=np.float)
    return tuple(np.array(arg, dtype=np.float) for arg in args)


def to_np_vector(x, by_column=True, thin_interval=None):
    if not isinstance(x, (torch.Tensor, np.ndarray)):
        raise ValueError('Invalid input type: {}'.format(type(x)))
    if isinstance(x, torch.Tensor):
        x = x.detach.numpy()
    if by_column:
        order = 'F'
    else:
        order = 'C' # by row
    np_vect = x.reshape(-1, order=order)
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


def get_wlabel(i=None, wlabel=W):
    return wlabel if i is None else wlabel + str(i)
