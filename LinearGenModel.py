import numpy as np
import pandas as pd
import torch
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from torch import nn
from scipy import stats
import warnings


from types import FunctionType, MethodType

from utils import to_np_vector, to_np_vectors, get_num_positional_args, W, T, Y
from plotting import compare_joints, compare_bivariate_marginals

import os
import sys


T_SITE = T + '_obs'
Y_SITE = Y + '_obs'

# N_SAMPLES = 100
N_SAMPLES_PER_W = 10

MODEL_LABEL = 'model'
TRUE_LABEL = 'true'
T_MODEL_LABEL = '{} ({})'.format(T, MODEL_LABEL)
Y_MODEL_LABEL = '{} ({})'.format(Y, MODEL_LABEL)
T_TRUE_LABEL = '{} ({})'.format(T, TRUE_LABEL)
Y_TRUE_LABEL = '{} ({})'.format(Y, TRUE_LABEL)
NAME = 'LinearGenModel'



class DataGenModel:

    def __init__(self, data, lambda0_t_w=1e-5, lambda0_y_tw=1e-5):
        # TODO: docstring that specifies
        # 1. data must be in w,t,y format
        self.data = self._matricize(data)
        self._train(lambda0_t_w, lambda0_y_tw)


    def _matricize(self, data):
        return [np.reshape(d, [d.shape[0], -1]) for d in data]


    def _train(self, lambda0_t_w=1e-5, lambda0_y_tw=1e-5):
        # fitting a linear Gaussian model defined as
        # p(y | x) = N(y; beta*x, sigma^2)
        # with a prior p(beta) = N(0, sigma^2/lambda0) which amounts to L2 penalty on beta: lambda||beta||^2
        # where x and y are input and output respectively
        # TODO: learning different sigma for each output feature

        w, t, y = self.data

        # t | w
        self.beta_t_w, self.sigma_t_w = self.linear_gaussian_solver(w, t, lambda0_t_w)

        # y | t and w
        self.beta_y_tw, self.sigma_y_tw = self.linear_gaussian_solver(np.concatenate([w, t], 1), y, lambda0_y_tw)


    def _pad_with_ones(self, X):
      return np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)


    def _linear_gaussian_solver(self, X, Y, lambda0=1e-5):
      beta = np.linalg.inv(X.T.dot(X) + lambda0 * np.identity(X.shape[1])).dot(X.T.dot(Y))
      var = ((Y - X.dot(beta)) ** 2).mean()
      # assuming sharing the save variance parameter across different yj's
      return beta, np.sqrt(var)


    def linear_gaussian_solver(self, X, Y, lambda0=1e-5):
        return self._linear_gaussian_solver(self._pad_with_ones(X), Y, lambda0)


    def gaussian_sampler(self, mean, sigma):
        return np.random.randn(*mean.shape) * sigma + mean

    def linear_gaussian_sampler(self, X, beta, sigma):
        mean = self._pad_with_ones(X).dot(beta)
        return self.gaussian_sampler(mean, sigma)


    def sample(self, n_samples_per_w=N_SAMPLES_PER_W, w_samples=None):

        # sample t and y given w
        # todo: more features


        if w_samples is None:
            w_samples = self.data[0]  # use "training data"

        t_samples = self.linear_gaussian_sampler(
            w_samples,
            self.beta_t_w,
            self.sigma_t_w
        )
        y_samples = self.linear_gaussian_sampler(
            np.concatenate([w_samples, t_samples], 1),
            self.beta_y_tw,
            self.sigma_y_tw
        )

        return w_samples, t_samples, y_samples


    def sample_interventional(self, intervention_val, site=T_SITE, **kwargs):
        pass

    def get_interventional_mean(self, intervention_val, treatment_site=T_SITE, outcome_site=Y_SITE, **kwargs):
        pass

    def get_ate(self, intervention_val1=0, intervention_val2=1, **kwargs):
        pass

    def plot_ty_dists(self, joint=True, marginal_hist=True, marginal_qq=True, name=NAME,
                      file_ext='pdf', n_samples_per_w=N_SAMPLES_PER_W, thin_model=None,
                      thin_true=None, t_site=T_SITE, y_site=Y_SITE, joint_kwargs={}, test=False):
        pass


if __name__ == '__main__':
    from data.synthetic import generate_wty_linear_scalar_data, generate_wty_linear_multi_w_data
    from utils import NUMPY
    data = generate_wty_linear_multi_w_data(500, data_format=NUMPY, wdim=5)

    dgm = DataGenModel(data)
    data_samples = dgm.sample()

    # TODO: do tests on the samples, visualize etc.