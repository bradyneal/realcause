import numpy as np
from models.base import BaseGenModel
import torch
from torch import nn
from torch.utils import data
from itertools import chain


Log2PI = float(np.log(2 * np.pi))
def log_normal(x, mean, log_var, eps=0.00001):
  z = - 0.5 * Log2PI
  return - (x - mean) ** 2 / (2. * torch.exp(log_var) + eps) - log_var / 2. + z


class MLPParams:
    def __init__(self, n_hidden_layers=1, dim_h=64, activation=nn.ReLU()):
        self.n_hidden_layers = n_hidden_layers
        self.dim_h = dim_h
        self.activation = activation


class TrainingParams:
    def __init__(self, batch_size=32, lr=0.001, num_epochs=100, verbose=True, print_every_iters=100,
                 optim=torch.optim.Adam, **optim_args):
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.print_every_iters = print_every_iters
        self.optim = optim
        self.optim_args = optim_args


class CausalDataset(data.Dataset):
    def __init__(self, w, t, y, wtype='float32', ttype='float32', ytype='float32'):
        self.w = w.astype(wtype)
        self.t = t.astype(ttype)
        self.y = y.astype(ytype)

    def __len__(self):
        return self.w.shape[0]

    def __getitem__(self, index):
        return self.w[index], self.t[index], self.y[index]


# TODO: for more complex w, we might need to share parameters (dependent on the problem)
class MLP(BaseGenModel):

    def __init__(self, w, t, y,
                 mlp_params_t_w=MLPParams(),
                 mlp_params_y_tw=MLPParams(),
                 training_params = TrainingParams(),
                 binary_treatment=False):
        super(MLP, self).__init__(*self._matricize((w, t, y)))
        self.binary_treatment = binary_treatment

        self.dim_w = self.w.shape[1]
        self.dim_t = self.t.shape[1]
        self.dim_y = self.y.shape[1]

        self.MLP_params_t_w = mlp_params_t_w
        self.MLP_params_y_tw = mlp_params_y_tw
        self.mlp_t_w = self._build_mlp(self.dim_w, self.dim_t, mlp_params_t_w)
        self.mlp_y_tw = self._build_mlp(self.dim_w+self.dim_t, self.dim_y, mlp_params_y_tw)

        # TODO: multiple optimizers ?
        self.training_params = training_params
        self.optim = training_params.optim(
            chain(self.mlp_t_w.parameters(), self.mlp_y_tw.parameters()),
            training_params.lr, **training_params.optim_args
        )

        # TODO: binary treatment -> long data type
        self.data_loader = data.DataLoader(CausalDataset(self.w, self.t, self.y),
                                           batch_size=training_params.batch_size)

        self._train()

    def _matricize(self, data):
        return [np.reshape(d, [d.shape[0], -1]) for d in data]

    def _build_mlp(self, dim_x, dim_y, MLP_params=MLPParams()):
        dim_h = MLP_params.dim_h
        hidden_layers = [nn.Linear(dim_x, dim_h), MLP_params.activation]
        for _ in range(MLP_params.n_hidden_layers-1):
            hidden_layers += [nn.Linear(dim_h, dim_h), MLP_params.activation]

        # heteroscedastic Gaussian
        hidden_layers += [nn.Linear(dim_h, dim_y * 2)]
        # TODO: binary data

        return nn.Sequential(*hidden_layers)

    def _train(self):
        c = 0
        for _ in range(self.training_params.num_epochs):
            for w, t, y in self.data_loader:
                self.optim.zero_grad()
                t_ = self.mlp_t_w(w)
                y_ = self.mlp_y_tw(torch.cat([w,t], dim=1))
                loss_t = - log_normal(t, *torch.chunk(t_, chunks=2, dim=1)).sum(1).mean()
                loss_y = - log_normal(y, *torch.chunk(y_, chunks=2, dim=1)).sum(1).mean()
                loss = loss_t + loss_y
                # TODO: learning rate can be separately adjusted by weighting the losses here
                loss.backward()
                self.optim.step()

                if self.training_params.verbose and c % self.training_params.print_every_iters == 0:
                    print("Iteration {}: {}".format(c, loss))
                c += 1

    def gaussian_sampler(self, mean, log_var):
        sigma = torch.exp(0.5*log_var)
        return torch.randn(*mean.shape) * sigma + mean

    def sample_t(self, w=None):
        if w is None:
            w = self.sample_w()
        t_ = self.mlp_t_w(torch.from_numpy(w).float())
        t_samples = self.gaussian_sampler(*torch.chunk(t_, chunks=2, dim=1)).data.cpu().numpy()
        if self.binary_treatment:
            t_samples = t_samples > .5
        return t_samples

    def sample_y(self, t, w=None):
        if w is None:
            w = self.sample_w()
        wt = np.concatenate([w, t], 1)
        y_ = self.mlp_y_tw(torch.from_numpy(wt).float())
        y_samples = self.gaussian_sampler(*torch.chunk(y_, chunks=2, dim=1)).data.cpu().numpy()
        return y_samples


if __name__ == '__main__':
    from data.synthetic import generate_wty_linear_multi_w_data
    from utils import NUMPY

    # data = generate_wty_linear_multi_w_data(500, data_format=NUMPY, wdim=5)
    #
    # dgm = DataGenModel(data)
    # data_samples = dgm.sample()

    w, t, y = generate_wty_linear_multi_w_data(500, data_format=NUMPY, wdim=5)

    mlp = MLP(w, t, y)
    data_samples = mlp.sample()
    mlp.plot_ty_dists()
    # uni_metrics = lgm.get_univariate_quant_metrics()
    # multi_ty_metrics = lgm.get_multivariate_quant_metrics(include_w=False)
    # multi_wty_metrics = lgm.get_multivariate_quant_metrics(include_w=True)
