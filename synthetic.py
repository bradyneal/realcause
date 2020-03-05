import numpy as np
import torch


def tfloat(*args):
    if len(args) == 1:
        return torch.tensor(args[0], dtype=torch.float)
    else:
        return tuple(torch.tensor(arg, dtype=torch.float) for arg in args)


def generate_xy_linear_scalar_data(n, w=5, x_mean=0, x_std=1, noise_mean=0, noise_std=None):
    if noise_std is None:
        noise_std = w
    x = np.random.normal(loc=x_mean, scale=x_std, size=n)
    y = (w * x) + np.random.normal(loc=noise_mean, scale=noise_std, size=n)
    return tfloat(x, y)


def generate_zty_linear_scalar_data(n, alpha=2, beta=10, delta=5, z_mean=0, z_std=1,
        t_noise_mean=0, t_noise_std=1, y_noise_mean=0, y_noise_std=1):
    z = np.random.normal(loc=z_mean, scale=z_std, size=n)
    t = (alpha * z) + np.random.normal(loc=t_noise_mean, scale=t_noise_std, size=n)
    y = (delta * t) + (beta * z) + np.random.normal(loc=y_noise_mean, scale=y_noise_std, size=n)
    return tfloat(z, t, y)