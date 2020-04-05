import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample, PyroParam
import torch
from torch import nn

# Note: below, w denotes the confounder, t denotes the treatment, and z denotes the latent variable for generation
class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, w_dim, t_dim=1, y_dim=1):
        super().__init__()
        assert z_dim > 1 and w_dim > 1 and t_dim == 1 and y_dim == 1
        self.z_to_h = nn.Linear(z_dim + w_dim + t_dim, hidden_dim)
        self.h_to_ymean = nn.Linear(hidden_dim, y_dim)
        self.h_to_ystd = nn.Linear(hidden_dim, y_dim)
        self.activation = nn.Softplus()

    def forward(self, z, w, t):
        assert z.ndim == 2
        assert w.ndim == 2
        assert t.ndim == 1
        zwt = torch.cat((z, w, t.unsqueeze(dim=1)), dim=1)
        hidden = self.activation(self.z_to_h(zwt))
        y_mean = self.h_to_ymean(hidden)
        y_std = torch.abs(self.h_to_ystd(hidden))
        y_std = torch.min(y_std, torch.Tensor([100]))
        # print('y_std\tmin: {} ... max {}'.format(y_std.min(), y_std.max()))

        # y_std = torch.ones(y_std.shape[0])
        return y_mean, y_std


class Encoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, w_dim, t_dim):
        super().__init__()
        assert z_dim > 1 and t_dim == 1
        self.y_to_h = nn.Linear(1, hidden_dim)
        self.h_to_zmean = nn.Linear(hidden_dim, z_dim)
        self.h_to_zstd = nn.Linear(hidden_dim, z_dim)
        # self.h_to_wmean = nn.Linear(hidden_dim, w_dim)
        # self.h_to_wstd = nn.Linear(hidden_dim, w_dim)
        # self.h_to_tmean = nn.Linear(hidden_dim, t_dim)
        # self.h_to_tstd = nn.Linear(hidden_dim, t_dim)
        self.activation = nn.Softplus()

    def forward(self, y):
        assert y.ndim == 1
        hidden = self.activation(self.y_to_h(y.unsqueeze(dim=1)))

        z_mean = self.h_to_zmean(hidden)
        z_std = torch.abs(self.h_to_zstd(hidden))
        z_std = torch.min(z_std, torch.Tensor([100]))
        # print('z_std\tmin: {} ... max {}'.format(z_std.min(), z_std.max()))

        # w_mean = self.h_to_wmean(hidden)
        # w_std = torch.exp(self.h_to_wstd(hidden))
        #
        # t_mean = self.h_to_tmean(hidden)
        # t_std = torch.exp(self.h_to_tstd(hidden))

        # return (z_mean, z_std), (w_mean, w_std), (t_mean, t_std)

        return z_mean, z_std


# TODO: VAE class is currently copy and pasted
class VAE(nn.Module):
    # by default our latent space is 50-dimensional
    # and we use 400 hidden units
    def __init__(self, hidden_dim=400, z_dim=50, w_dim=10, t_dim=1, y_dim=1, seed=0, use_cuda=False):
        super().__init__()
        pyro.set_rng_seed(seed)

        # create the encoder and decoder networks
        self.encoder = Encoder(hidden_dim=hidden_dim, z_dim=z_dim, w_dim=w_dim, t_dim=t_dim)
        self.decoder = Decoder(hidden_dim=hidden_dim, z_dim=z_dim, w_dim=w_dim, t_dim=t_dim, y_dim=y_dim)

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
        self.use_cuda = use_cuda
        self.z_dim = z_dim

    # define the model p(x|z)p(z)
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
            y_mean, y_std = self.decoder.forward(z, w, t)
            # score against actual images
            pyro.sample("y_obs", dist.Normal(y_mean, y_std).to_event(1), obs=y)

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, w, t, y):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", w.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            (z_mean, z_std) = self.encoder.forward(y)
            # sample the latent code z
            pyro.sample("latent", dist.Normal(z_mean, z_std).to_event(1))

    # # define a helper function for reconstructing images
    # def reconstruct_img(self, x):
    #     # encode image x
    #     z_loc, z_scale = self.encoder(x)
    #     # sample in latent space
    #     z = dist.Normal(z_loc, z_scale).sample()
    #     # decode the image (note we don't sample in image space)
    #     loc_img = self.decoder(z)
    #     return loc_img
