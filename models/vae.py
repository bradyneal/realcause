import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample, PyroParam
import torch
from torch import nn
import torch.nn.functional as F


# TODO: maybe try a linear VAE
# Note: below, w denotes the confounder, t denotes the treatment, and z denotes the latent variable for generation
class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, y_dim=1, debug=False):
        super().__init__()
        self.z_to_h = nn.Linear(z_dim, hidden_dim)
        self.h_to_ymean = nn.Linear(hidden_dim, y_dim)
        self.h_to_ystd = nn.Linear(hidden_dim, y_dim)
        self.activation = nn.ReLU()
        # self.activation = nn.Softplus()
        self.debug = debug

    def forward(self, z):
        hidden = self.activation(self.z_to_h(z))
        y_mean = self.h_to_ymean(hidden)
        y_std = F.softplus(self.h_to_ystd(hidden))

        if self.debug:
            print('y_mean\tmin: {} ... max {}'.format(y_mean.min(), y_mean.max()))
            print('y_std\tmin: {} ... max {}'.format(y_std.min(), y_std.max()))
        return y_mean, y_std


class Encoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, y_dim, debug=False):
        super().__init__()
        self.y_to_h = nn.Linear(y_dim, hidden_dim)
        self.h_to_zmean = nn.Linear(hidden_dim, z_dim)
        self.h_to_zstd = nn.Linear(hidden_dim, z_dim)
        self.activation = nn.ReLU()
        # self.activation = nn.Softplus()
        self.debug = debug

    def forward(self, y):
        hidden = self.activation(self.y_to_h(y))
        z_mean = self.h_to_zmean(hidden)
        z_std = F.softplus(self.h_to_zstd(hidden))

        if self.debug:
            print('z_mean\tmin: {} ... max {}'.format(z_mean.min(), z_mean.max()))
            print('z_std\tmin: {} ... max {}'.format(z_std.min(), z_std.max()))
        return z_mean, z_std


# NOTE: VAE class is mostly copy and pasted from Pyro VAE tutorial
class VAE(nn.Module):
    def __init__(self, hidden_dim=400, z_dim=50, w_dim=10, t_dim=1, y_dim=1, seed=0, use_cuda=False, debug=False):
        super().__init__()
        pyro.set_rng_seed(seed)

        # create the encoder and decoder networks
        ywt_dim = y_dim + w_dim + t_dim
        zwt_dim = z_dim + w_dim + t_dim
        self.encoder = Encoder(y_dim=ywt_dim, hidden_dim=hidden_dim, z_dim=z_dim, debug=debug)
        self.decoder = Decoder(z_dim=zwt_dim, hidden_dim=hidden_dim, y_dim=y_dim, debug=debug)

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
        self.use_cuda = use_cuda
        self.z_dim = z_dim

    # define the model p(y|z,w,t)p(z,w,t) = p(y|z,w,t)p(z)
    def model(self, w, t, y=None):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", w.shape[0]):
            # setup hyperparameters for prior p(z)
            z_loc = w.new_zeros(torch.Size((w.shape[0], self.z_dim)))
            z_scale = w.new_ones(torch.Size((w.shape[0], self.z_dim)))
            # sample from prior (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            # decode the latent code z
            assert z.ndim == 2 and w.ndim == 2 and t.ndim == 1
            zwt = torch.cat((z, w, t.unsqueeze(dim=1)), dim=1)
            y_mean, y_std = self.decoder.forward(zwt)
            # score against actual y
            pyro.sample("y_obs", dist.Normal(y_mean, y_std).to_event(1), obs=y)

    # define the guide (i.e. variational distribution) q(z|y,w,t)
    def guide(self, w, t, y=None):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", w.shape[0]):
            if y is None:   # Evaluation mode for using Pyro Predictive class
                z_loc = w.new_zeros(torch.Size((w.shape[0], self.z_dim)))
                z_scale = w.new_ones(torch.Size((w.shape[0], self.z_dim)))
                pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            else:           # Regular training model
                # use the encoder to get the parameters used to define q(z|y,w,t)
                assert y.ndim == 1 and w.ndim == 2 and t.ndim == 1
                ywt = torch.cat((y.unsqueeze(dim=1), w, t.unsqueeze(dim=1)), dim=1)
                (z_mean, z_std) = self.encoder.forward(ywt)
                # sample the latent code z
                pyro.sample("latent", dist.Normal(z_mean, z_std).to_event(1))
