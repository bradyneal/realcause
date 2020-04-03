import numpy as np
from utils import to_data_format, PANDAS, TORCH


def generate_zty_linear_scalar_data(n, data_format=PANDAS, seed=0, alpha=-2, beta=2, delta=2,
                                    z_mean=0, z_std=1, t_noise_mean=0, t_noise_std=1,
                                    y_noise_mean=0, y_noise_std=1):
    np.random.seed(seed)
    z = np.random.normal(loc=z_mean, scale=z_std, size=n)
    t = (alpha * z) + np.random.normal(loc=t_noise_mean, scale=t_noise_std, size=n)
    y = (delta * t) + (beta * z) + np.random.normal(loc=y_noise_mean, scale=y_noise_std, size=n)
    return to_data_format(data_format, z, t, y)


def generate_zty_linear_multi_z_data(n, zdim=10, data_format=TORCH, seed=0, delta=2,
                                     alpha_mean=0, alpha_std=1, beta_mean=0, beta_std=1,
                                     z_mean=0, z_std=1, t_noise_mean=0, t_noise_std=1,
                                     y_noise_mean=0, y_noise_std=1):
        np.random.seed(seed)
        z = np.random.normal(loc=z_mean, scale=z_std, size=(n, zdim))
        alpha = np.random.normal(loc=alpha_mean, scale=alpha_std, size=(zdim, 1))
        beta = np.random.normal(loc=beta_mean, scale=beta_std, size=(zdim, 1))
        t = (z @ alpha).squeeze() + np.random.normal(loc=t_noise_mean, scale=t_noise_std, size=n)
        y = (delta * t) + (z @ beta).squeeze() + np.random.normal(loc=y_noise_mean, scale=y_noise_std, size=n)
        return to_data_format(data_format, z, t, y)
