import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample, PyroParam
import torch
from torch import nn


def linear_gaussian_full_model(w, t=None, y=None):
    sigma_t = pyro.sample("sigma_t", dist.Uniform(0., 10.))
    w_wt = pyro.sample('w_wt', dist.Normal(0., 10.))
    b_t = pyro.sample('b_t', dist.Normal(0., 10.))

    sigma_y = pyro.sample('sigma_y', dist.Uniform(0., 10.))
    w_ty = pyro.sample('w_ty', dist.Normal(0., 10.))
    w_wy = pyro.sample('w_wy', dist.Normal(0., 10.))
    b_y = pyro.sample('b_y', dist.Normal(0., 10.))

    with pyro.plate("data", w.shape[0]):
        t = pyro.sample("t_obs", dist.Normal(w_wt * w + b_t, sigma_t), obs=t)
        y = pyro.sample("y_obs", dist.Normal(w_ty * t + w_wy * w + b_y, sigma_y), obs=y)

    return t, y


def linear_gaussian_assignment_model(w, t=None):
    sigma_t = pyro.sample("sigma_t", dist.Uniform(0., 10.))
    w_wt = pyro.sample('w_wt', dist.Normal(0., 10.))
    b_t = pyro.sample('b_t', dist.Normal(0., 10.))

    with pyro.plate("data", w.shape[0]):
        t = pyro.sample("t_obs", dist.Normal(w_wt * w + b_t, sigma_t), obs=t)

    return t


def linear_gaussian_outcome_model(w, t, y=None):
    sigma_y = pyro.sample('sigma_y', dist.Uniform(0., 10.))
    w_ty = pyro.sample('w_ty', dist.Normal(0., 10.))
    w_wy = pyro.sample('w_wy', dist.Normal(0., 10.))
    b_y = pyro.sample('b_y', dist.Normal(0., 10.))

    with pyro.plate("data", w.shape[0]):
        y = pyro.sample("y_obs", dist.Normal(w_ty * t + w_wy * w + b_y, sigma_y), obs=y)

    return y


def linear_multi_w_outcome_model(w, t, y=None):
    assert w.ndim == 2 and w.shape[1] > 1

    class LinearGaussianOutcomeModel(PyroModule):
        def __init__(self, wdim):
            super().__init__()

            self.linear_wy = PyroModule[nn.Linear](wdim, 1)
            self.linear_wy.weight = PyroSample(dist.Normal(0., 10.).expand([1, wdim]).to_event(2))
            self.linear_wy.bias = PyroSample(dist.Normal(0., 10.))

        def forward(self, w, t, y=None):
            w_ty = pyro.sample('w_ty', dist.Normal(0., 10.))
            y_mean = self.linear_wy(w).squeeze(-1) + w_ty * t

            sigma_y = pyro.sample("sigma_y", dist.Uniform(0., 10.))
            with pyro.plate("data", w.shape[0]):
                y = pyro.sample("y_obs", dist.Normal(y_mean, sigma_y), obs=y)

            return y

    return LinearGaussianOutcomeModel(w.shape[1])(w, t, y)
