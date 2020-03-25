import numpy as np
import torch
import pandas as pd
from DataGenModel import Z, T, Y


def generate_zty_linear_scalar_data(n, format='pandas', seed=0, alpha=-2, beta=2, delta=2,
                                    z_mean=0, z_std=1, t_noise_mean=0, t_noise_std=1,
                                    y_noise_mean=0, y_noise_std=1):
    np.random.seed(seed)
    z = np.random.normal(loc=z_mean, scale=z_std, size=n)
    t = (alpha * z) + np.random.normal(loc=t_noise_mean, scale=t_noise_std, size=n)
    y = (delta * t) + (beta * z) + np.random.normal(loc=y_noise_mean, scale=y_noise_std, size=n)
    if format.lower() == 'pandas':
        return pd.DataFrame({Z: z, T: t, Y: y})
    elif format.lower() == 'torch':
        return tfloat(z, t, y)


def tfloat(*args):
    if len(args) == 1:
        return torch.tensor(args[0], dtype=torch.float)
    else:
        return tuple(torch.tensor(arg, dtype=torch.float) for arg in args)
