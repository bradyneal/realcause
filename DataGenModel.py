import numpy as np
import pandas as pd
import torch
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from torch import nn
from scipy import stats

import pyro
import pyro.optim
import pyro.distributions as dist
# from pyro.nn import PyroModule, PyroSample
from pyro.infer.autoguide import AutoDiagonalNormal, AutoMultivariateNormal, AutoDelta, AutoNormal
from pyro.infer import SVI, Trace_ELBO, Predictive
from torch.distributions import constraints

from types import FunctionType

from util import to_np_vector, to_np_vectors
from plotting import compare_joints, compare_bivariate_marginals

Z = 'z'
T = 't'
Y = 'y'
T_SITE = T + '_obs'
Y_SITE = Y + '_obs'
MCMC = 'NUTS'
# N_SAMPLES = 100
N_SAMPLES_PER_Z = 10

MODEL_LABEL = 'model'
TRUE_LABEL = 'true'
T_MODEL_LABEL = '{} ({})'.format(T, MODEL_LABEL)
Y_MODEL_LABEL = '{} ({})'.format(Y, MODEL_LABEL)
T_TRUE_LABEL = '{} ({})'.format(T, TRUE_LABEL)
Y_TRUE_LABEL = '{} ({})'.format(Y, TRUE_LABEL)
SAVE_NAME = 'DataGenModel'


class DataGenModel:

    def __init__(self, data, model, guide, svi=True, opt=None, lr=0.03, n_iters=5000, log_interval=100, mcmc=None,
                 col_labels={Z: Z, T: T, Y: Y}, seed=0, enable_validation=True):
        # TODO: docstring that specifies
        # 1. data must be in z,t,y format
        # 2. model is a pyro model and guide is a matching pyro guide or autoguide
        pyro.set_rng_seed(seed)
        pyro.enable_validation(enable_validation)  # Good for debugging

        self.data = data
        self.model = model
        self.guide = guide
        self.mcmc = mcmc
        self.zlabel = col_labels[Z]
        self.tlabel = col_labels[T]
        self.ylabel = col_labels[Y]

        # Prepare guide
        # "Meta-guide" that takes model as argument to get actual guide
        if isinstance(guide, pyro.nn.module._PyroModuleMeta):
            self.guide = guide(model)
        # Hopefully, a regular guide function
        elif isinstance(self.guide, FunctionType):
            self.guide = guide
        else:
            raise ValueError('Invalid guide: {}'.format(self.guide))

        # Prepare posterior
        if svi and mcmc is not None:
            raise ValueError('Cannot do both SVI and MCMC. Choose one.')
        if svi:
            if opt is None:
                opt = pyro.optim.Adam({'lr': lr})
            self._train(opt, n_iters, log_interval)
        else:
            if mcmc is None:
                self.mcmc = MCMC
            else:
                self.mcmc = mcmc

    def _train(self, opt, n_iters, log_interval):
        svi = SVI(self.model, self.guide, opt, loss=Trace_ELBO())
        pyro.clear_param_store()
        for i in range(n_iters):
            # calculate the loss and take a gradient step
            z, t, y = self._get_data_tensors([self.zlabel, self.tlabel, self.ylabel])
            loss = svi.step(z, t, y)
            if i % log_interval == 0:
                print("[iter %04d] loss: %.4f" % (i + 1, loss / len(y)))

    def _get_data_tensors(self, labels):
        """
        Get the torch tensors corresponding to the columns with the specified labels
        :param labels: list/tuple of column name(s) of self.data, a pandas DataFrame
        :return: tuple of torch tensors (one for each label)
        """
        if not (isinstance(labels, str) or isinstance(labels, list) or isinstance(labels, tuple)):
            raise ValueError('Invalid input: {}'.format(labels))
        if isinstance(labels, str):
            labels = (labels,)
        tensors = tuple(torch.tensor(self.data[label], dtype=torch.float) for label in labels)
        if len(tensors) == 1:
            return tensors[0]
        else:
            return tensors

    def sample(self, n_samples_per_z=N_SAMPLES_PER_Z, z_samples=None, sites=(T_SITE, Y_SITE)):
        # NOTE: might need to use 'posterior_samples' arg for MCMC
        if z_samples is None:
            z = self._get_data_tensors(self.zlabel)     # use "training data"
        pred = Predictive(self.model, guide=self.guide, num_samples=n_samples_per_z, return_sites=sites)
        samples = pred(z)
        return samples

    def plot_ty_dists(self, joint=True, marginal_hist=True, marginal_qq=True, save_name=SAVE_NAME, n_samples_per_z=N_SAMPLES_PER_Z,
                      thin_model=None, thin_true=None, t_site=T_SITE, y_site=Y_SITE, joint_kwargs={}):
        samples = self.sample(n_samples_per_z, sites=(t_site, y_site))
        t_model, y_model = to_np_vectors([samples[T_SITE], samples[Y_SITE]], thin_interval=thin_model)
        t_true, y_true = to_np_vectors(self._get_data_tensors([self.tlabel, self.ylabel]), thin_interval=thin_true)

        if joint:
            compare_joints(t_model, y_model, t_true, y_true,
                           xlabel1=T_MODEL_LABEL, ylabel1=Y_MODEL_LABEL,
                           xlabel2=T_TRUE_LABEL, ylabel2=Y_TRUE_LABEL,
                           save_fname='{}_ty_joints.pdf'.format(save_name),
                           kwargs=joint_kwargs)

        if marginal_hist or marginal_qq:
            compare_bivariate_marginals(t_model, t_true, y_model, y_true,
                                        xlabel=T, ylabel=Y,
                                        label1=MODEL_LABEL, label2=TRUE_LABEL,
                                        hist=marginal_hist, qqplot=marginal_qq,
                                        save_hist_fname='{}_ty_marginal_hists.pdf'.format(save_name),
                                        save_qq_fname='{}_ty_marginal_qqplots.pdf'.format(save_name))

    # TODO: implement holding out data and evaluating stuff (e.g. log-likelihood and quant diag) on that
