import numpy as np
from utils import to_data_format, NUMPY


def generate_wty_linear_scalar_data(n, data_format=NUMPY, binary_treatment=False,
                                    w_2d=False, seed=0, alpha=-2, beta=2, delta=2,
                                    w_mean=0, w_std=1, t_noise_mean=0, t_noise_std=1,
                                    y_noise_mean=0, y_noise_std=1):
    np.random.seed(seed)
    if w_2d:
        size = (n, 1)
    else:
        size = n
    w = np.random.normal(loc=w_mean, scale=w_std, size=size)
    t = (alpha * w) + np.random.normal(loc=t_noise_mean, scale=t_noise_std, size=size)
    if binary_treatment:
        t = t > 0
    y = (delta * t) + (beta * w) + np.random.normal(loc=y_noise_mean, scale=y_noise_std, size=size)
    return to_data_format(data_format, w, t, y)


def generate_wty_linear_multi_w_data(n, wdim=10, data_format=NUMPY,
                                     binary_treatment=False, seed=0, delta=2,
                                     alpha_mean=0, alpha_std=1, beta_mean=0, beta_std=1,
                                     w_mean=0, w_std=1, t_noise_mean=0, t_noise_std=1,
                                     y_noise_mean=0, y_noise_std=1):
    np.random.seed(seed)
    w = np.random.normal(loc=w_mean, scale=w_std, size=(n, wdim))
    alpha = np.random.normal(loc=alpha_mean, scale=alpha_std, size=(wdim, 1))
    beta = np.random.normal(loc=beta_mean, scale=beta_std, size=(wdim, 1))
    t = (w @ alpha).squeeze() + np.random.normal(loc=t_noise_mean, scale=t_noise_std, size=n)
    if binary_treatment:
        t = t > 0
    y = (delta * t) + (w @ beta).squeeze() + np.random.normal(loc=y_noise_mean, scale=y_noise_std, size=n)
    return to_data_format(data_format, w, t, y)
