import numpy as np
import pandas as pd
import torch
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from torch import nn
from scipy import stats
import warnings

import pyro
import pyro.optim
import pyro.distributions as dist
# from pyro.nn import PyroModule, PyroSample
from pyro.infer.autoguide import AutoDiagonalNormal, AutoMultivariateNormal, AutoDelta, AutoNormal
from pyro.infer import SVI, Trace_ELBO, Predictive, MCMC, NUTS, HMC
from pyro.infer.mcmc.mcmc_kernel import MCMCKernel
from torch.distributions import constraints
from pyro.infer.autoguide import AutoGuide

from types import FunctionType, MethodType

from utils import to_np_vector, to_np_vectors, get_num_positional_args, Z, T, Y
from plotting import compare_joints, compare_bivariate_marginals

import os
import sys

str_to_mcmc_sampler = {
    'NUTS': NUTS,
    'HMC': HMC,
}

T_SITE = T + '_obs'
Y_SITE = Y + '_obs'
MCMC_DEFAULT = NUTS
# N_SAMPLES = 100
N_SAMPLES_PER_Z = 10

MODEL_LABEL = 'model'
TRUE_LABEL = 'true'
T_MODEL_LABEL = '{} ({})'.format(T, MODEL_LABEL)
Y_MODEL_LABEL = '{} ({})'.format(Y, MODEL_LABEL)
T_TRUE_LABEL = '{} ({})'.format(T, TRUE_LABEL)
Y_TRUE_LABEL = '{} ({})'.format(Y, TRUE_LABEL)
NAME = 'DataGenModel'


class DataGenModel:

    def __init__(self, data, model, guide=None, svi=True, opt=None, lr=0.03, n_iters=5000, log_interval=100, mcmc=None,
                 col_labels={Z: Z, T: T, Y: Y}, seed=0, enable_validation=True):
        # TODO: docstring that specifies
        # 1. data must be in z,t,y format
        # 2. model is a pyro model and guide is a matching pyro guide or autoguide
        pyro.set_rng_seed(seed)
        pyro.enable_validation(enable_validation)  # Good for debugging

        self.data = data
        self.model = model
        self.svi = svi
        self.zlabel = col_labels[Z]
        self.tlabel = col_labels[T]
        self.ylabel = col_labels[Y]

        # Prepare posterior
        if svi and mcmc:
            raise ValueError('Cannot do both SVI and MCMC. Choose one.')
        if svi:
            # Prepare guide
            # "Meta-guide" that takes model as argument to get actual guide
            if guide is None:
                raise ValueError('A guide is a mandatory argument when using SVI. Please specify it.')
            elif isinstance(guide, pyro.nn.module._PyroModuleMeta):
                self.guide = guide(model)
            # Hopefully, a regular guide function
            elif isinstance(guide, (FunctionType, MethodType)):
                self.guide = guide
            else:
                raise ValueError('Invalid guide: {}'.format(guide))

            if opt is None:
                opt = pyro.optim.Adam({'lr': lr})

            self._train(opt, n_iters, log_interval)
        else:
            if mcmc is None:
                mcmc = MCMC_DEFAULT
            elif isinstance(mcmc, str) and mcmc.upper() in str_to_mcmc_sampler.keys():
                mcmc = str_to_mcmc_sampler[mcmc.upper()]

            if isinstance(mcmc, MCMCKernel):
                print('Using fully specified MCMC kernel')
                self.mcmc_kernel = mcmc
            elif mcmc is NUTS:
                print('Using NUTS default kernel')
                self.mcmc_kernel = NUTS(model)
            elif mcmc is HMC:
                raise ValueError('A default HMC is not supported. Please either use NUTS '
                                 'or give an HMC kernel that is fully specified.')
            else:
                raise ValueError('Unsupported mcmc: {}'.format(mcmc))

            if guide is not None:
                warnings.warn('the "guide" argument is currently not supported for MCMC, so it will be ignored.', Warning)

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
        :param labels: list/tuple of column name(s) of self.data, which is either a
            a Pandas DataFrame or a homogeneous list/tuple of ndarrays of torch.Tensors
        :return: tuple of torch tensors (one for each label)
        """
        if not isinstance(labels, (str, list, tuple)):
            raise ValueError('Invalid input: {}'.format(labels))
        if isinstance(labels, str):
            labels = (labels,)

        if isinstance(self.data, pd.DataFrame):
            tensors = tuple(torch.tensor(self.data[label], dtype=torch.float) for label in labels)
        elif isinstance(self.data, (list, tuple)):
            data_length = len(self.data)
            assert len(self.data) == data_length
            str_to_index = {
                'z': 0,
                't': 1,
                'y': 2
            }

            if all(isinstance(x, np.ndarray) for x in self.data):
                tensors = tuple(torch.tensor(self.data[str_to_index[label]], dtype=torch.float) for label in labels)
            elif all(isinstance(x, torch.Tensor) for x in self.data):
                tensors = tuple(self.data[str_to_index[label]] for label in labels)
            else:
                raise ValueError('Invalid data types in self.data: {}, {}, {}'.format(
                    (type(self.data[i]) for i in range(data_length))))
        else:
            raise ValueError('self.data is an invalid data type: {}'.format(type(self.data)))

        if len(tensors) == 1:
            return tensors[0]
        else:
            return tensors

    def sample(self, n_samples_per_z=N_SAMPLES_PER_Z, model=None, z_samples=None, t_samples=None, y_samples=None, sites=(T_SITE, Y_SITE), mcmc_kwargs={}):
        if model is None:
            model = self.model
        if self.svi:
            pred = Predictive(model, guide=self.guide, num_samples=n_samples_per_z, return_sites=sites)
        else:   # MCMC
            mcmc = MCMC(self.mcmc_kernel, num_samples=n_samples_per_z, **mcmc_kwargs)
            z, t, y = self._get_data_tensors([self.zlabel, self.tlabel, self.ylabel])
            mcmc.run(z, t, y)
            pred = Predictive(model, posterior_samples=mcmc.get_samples(),
                              num_samples=n_samples_per_z, return_sites=sites)

            # To avoid OMP error that results from mcmc.summary() not playing nicely with matplotlib on MacOS
            # Remove condition if it occurs on other operating systems
            if sys.platform == "darwin":
                os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

            print('\n\nMCMC posterior summary:')
            mcmc.summary()

        if z_samples is None:
            z_samples = self._get_data_tensors(self.zlabel)  # use "training data"

        # Decide between full models that only take z as input (only condition on z)
        # and outcome models that take both z and t as input (condition on both z and t)
        # and models that take z, t, and y as input (e.g. VAEs)
        n_positional_args = get_num_positional_args(self.model)
        if isinstance(self.guide, (FunctionType, MethodType)):
            n_positional_args = max(n_positional_args, get_num_positional_args(self.guide))
        elif isinstance(self.guide, AutoGuide) and isinstance(self.model, MethodType):
            warnings.warn('If using VAE and AutoGuide, it is unclear whether '
                          'DataGenModel.sample() currently works with these.')

        if n_positional_args == 1:
            samples = pred(z_samples)
        elif n_positional_args == 2:
            if t_samples is None:
                raise ValueError('t_samples argument must be specified, if the model takes 2 arguments (e.g. z and t)')
            if t_samples.shape[0] != z_samples.shape[0]:
                t_samples = t_samples.repeat_interleave(z_samples.shape[0])
            samples = pred(z_samples, t_samples)
        elif n_positional_args == 3:
            if y_samples is None:
                y_samples = self._get_data_tensors(self.ylabel)
            if t_samples.shape[0] != z_samples.shape[0]:
                t_samples = t_samples.repeat_interleave(z_samples.shape[0])
            samples = pred(z_samples, t_samples, y_samples)
        else:
            raise ValueError('model/guide has unsupported number of positional arguments:', n_positional_args)
        return samples

    def sample_interventional(self, intervention_val, site=T_SITE, **kwargs):
        if not isinstance(intervention_val, torch.Tensor):
            intervention_val = torch.tensor([intervention_val], dtype=torch.float32)
        interventional_model = pyro.do(self.model, data={site: intervention_val})
        return self.sample(model=interventional_model, t_samples=intervention_val, **kwargs)

    def get_interventional_mean(self, intervention_val, treatment_site=T_SITE, outcome_site=Y_SITE, **kwargs):
        samples = self.sample_interventional(intervention_val=intervention_val, site=treatment_site, **kwargs)
        return to_np_vector(samples[outcome_site]).mean()

    def get_ate(self, intervention_val1=0, intervention_val2=1, **kwargs):
        return self.get_interventional_mean(intervention_val2, **kwargs) - \
               self.get_interventional_mean(intervention_val1, **kwargs)

    def plot_ty_dists(self, joint=True, marginal_hist=True, marginal_qq=True, name=NAME, n_samples_per_z=N_SAMPLES_PER_Z,
                      thin_model=None, thin_true=None, t_site=T_SITE, y_site=Y_SITE, joint_kwargs={}):
        samples = self.sample(n_samples_per_z, sites=(t_site, y_site))
        t_model, y_model = to_np_vectors([samples[T_SITE], samples[Y_SITE]], thin_interval=thin_model)
        t_true, y_true = to_np_vectors(self._get_data_tensors([self.tlabel, self.ylabel]), thin_interval=thin_true)

        if joint:
            compare_joints(t_model, y_model, t_true, y_true,
                           xlabel1=T_MODEL_LABEL, ylabel1=Y_MODEL_LABEL,
                           xlabel2=T_TRUE_LABEL, ylabel2=Y_TRUE_LABEL,
                           save_fname='{}_ty_joints.pdf'.format(name),
                           name=name, kwargs=joint_kwargs)

        if marginal_hist or marginal_qq:
            compare_bivariate_marginals(t_true, t_model, y_true, y_model,
                                        xlabel=T, ylabel=Y,
                                        label1=TRUE_LABEL, label2=MODEL_LABEL,
                                        hist=marginal_hist, qqplot=marginal_qq,
                                        save_hist_fname='{}_ty_marginal_hists.pdf'.format(name),
                                        save_qq_fname='{}_ty_marginal_qqplots.pdf'.format(name),
                                        name=name)

    # TODO: implement holding out data and evaluating stuff (e.g. log-likelihood and quant diag) on that
