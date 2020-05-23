import numpy as np
from models.base import BaseGenModel
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from itertools import chain


Log2PI = float(np.log(2 * np.pi))
def log_normal(x, mean, log_var, eps=0.00001):
    z = - 0.5 * Log2PI
    return - (x - mean) ** 2 / (2. * torch.exp(log_var) + eps) - log_var / 2. + z


def log_log_logistic(x, log_alpha, log_beta_):
    """log density of log-logistic with alpha > 0 and beta > 1 (log beta > 0)
    log_beta_ is reparameterization of the shape parameter, beta = exp(softplus(log_beta_)) > 1
              which ensures the distribution has a well-defined expected value.
    see https://en.wikipedia.org/wiki/Log-logistic_distribution for more information
    support: x > 0
    """
    log_beta = F.softplus(log_beta_)
    beta = torch.exp(log_beta)
    log_x = torch.log(x)
    return log_beta + (beta - 1) * log_x - beta * log_alpha - 2 * F.softplus(beta * (log_x - log_alpha))


def log_mixed_log_logistic(x, logit_pi, log_alpha, log_beta_):
    """log likelihood of the following mixed distribution:
    with probability pi = sigmoid(logit_pi), x follows log-logistic(alpha, exp(softplut(log_beta_)))
    with probability 1-pi, x = 0
    """
    xp = x[x > 0]
    ll = torch.zeros_like(x)
    ll[x == 0] = F.logsigmoid(-logit_pi)[x == 0]
    ll[x > 0] = F.logsigmoid(logit_pi)[x > 0] + log_log_logistic(xp, log_alpha[x > 0], log_beta_[x > 0])
    return ll


def binary_cross_entropy(p, t):
    """p: prediction, t: target"""
    return F.binary_cross_entropy_with_logits(p, t)


def logistic_sampler(mu, s):
    """https://en.wikipedia.org/wiki/Logistic_distribution"""
    u = torch.rand_like(mu)
    x = torch.log(u) - torch.log(1-u)
    return mu + s * x


def log_logistic_sampler(log_alpha, log_beta_):
    """log-logistic with alpha > 0 and beta > 1 (log beta > 0)
    log_beta_ is reparameterization of the shape parameter, beta = exp(softplus(log_beta_)) > 1
    https://en.wikipedia.org/wiki/Log-logistic_distribution
    """
    return torch.exp(logistic_sampler(log_alpha, torch.exp(-F.softplus(log_beta_))))


def mixed_log_logistic_sampler(logit_pi, log_alpha, log_beta_):
    """
    with probability pi = sigmoid(logit_pi), x follows log-logistic(alpha, exp(softplut(log_beta_)))
    with probability 1-pi, x = 0
    """
    pi = torch.sigmoid(logit_pi)
    y = torch.zeros_like(pi)
    yp_ind = pi > torch.rand_like(pi)
    y[yp_ind] = log_logistic_sampler(log_alpha[yp_ind], log_beta_[yp_ind])
    return y

def gaussian_sampler(mean, log_var):
    sigma = torch.exp(0.5*log_var)
    return torch.randn(*mean.shape) * sigma + mean


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

    def __init__(self, w, t, y, seed=1,
                 mlp_params_t_w=MLPParams(),
                 mlp_params_y_tw=MLPParams(),
                 training_params = TrainingParams(),
                 binary_treatment=False,
                 outcome_distribution='gaussian',
                 outcome_min=None,
                 outcome_max=None):
        super(MLP, self).__init__(*self._matricize((w, t, y)), seed=seed)
        self.binary_treatment = binary_treatment
        self.outcome_distribution = outcome_distribution
        self.outcome_min = outcome_min
        self.outcome_max = outcome_max

        self.dim_w = self.w.shape[1]
        self.dim_t = self.t.shape[1]
        self.dim_y = self.y.shape[1]

        self.MLP_params_t_w = mlp_params_t_w
        self.MLP_params_y_tw = mlp_params_y_tw
        output_multiplier_t = 1 if binary_treatment else 2
        if outcome_distribution == 'gaussian':
            output_multiplier_y = 2
        elif outcome_distribution == 'log_logistic':
            output_multiplier_y = 2
        elif outcome_distribution == 'mixed_log_logistic':
            output_multiplier_y = 3
        else:
            raise('outcome distribution {} not implemented'.format(outcome_distribution))
        self.mlp_t_w = self._build_mlp(self.dim_w, self.dim_t, mlp_params_t_w, output_multiplier_t)
        self.mlp_y_tw = self._build_mlp(self.dim_w+self.dim_t, self.dim_y, mlp_params_y_tw, output_multiplier_y)

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

    def _build_mlp(self, dim_x, dim_y, MLP_params=MLPParams(), output_multiplier=2):
        dim_h = MLP_params.dim_h
        hidden_layers = [nn.Linear(dim_x, dim_h), MLP_params.activation]
        for _ in range(MLP_params.n_hidden_layers-1):
            hidden_layers += [nn.Linear(dim_h, dim_h), MLP_params.activation]
        hidden_layers += [nn.Linear(dim_h, dim_y * output_multiplier)]
        return nn.Sequential(*hidden_layers)

    def _treatment_loss(self, t_, t):
        if self.binary_treatment:
            loss_t = binary_cross_entropy(t_, t)
        else:
            loss_t = - log_normal(t, *torch.chunk(t_, chunks=2, dim=1)).sum(1).mean()
        return loss_t

    def _outcome_loss(self, y_, y):
        if self.outcome_distribution == 'log_logistic':
            log_alpha, log_beta_ = torch.chunk(y_, chunks=2, dim=1)
            loss_y = - log_log_logistic(y, log_alpha, log_beta_).mean()
        elif self.outcome_distribution == 'mixed_log_logistic':
            logit_pi, log_alpha, log_beta_ = torch.chunk(y_, chunks=3, dim=1)
            loss_y = - log_mixed_log_logistic(y, logit_pi, log_alpha, log_beta_).mean()
        elif self.outcome_distribution in ['gaussian', 'normal']:
            loss_y = - log_normal(y, *torch.chunk(y_, chunks=2, dim=1)).sum(1).mean()
        else:
            raise('outcome distribution {} not implemented'.format(self.outcome_distribution))
        return loss_y

    def _train(self):
        c = 0
        for _ in range(self.training_params.num_epochs):
            for w, t, y in self.data_loader:
                self.optim.zero_grad()
                t_ = self.mlp_t_w(w)
                y_ = self.mlp_y_tw(torch.cat([w,t], dim=1))
                loss_t = self._treatment_loss(t_, t)
                loss_y = self._outcome_loss(y_, y)
                loss = loss_t + loss_y
                # TODO: learning rate can be separately adjusted by weighting the losses here
                loss.backward()
                self.optim.step()

                if self.training_params.verbose and c % self.training_params.print_every_iters == 0:
                    print("Iteration {}: {}".format(c, loss))
                c += 1

    def sample_t(self, w=None):
        if w is None:
            w = self.sample_w()
        t_ = self.mlp_t_w(torch.from_numpy(w).float())
        if self.binary_treatment:
            t_samples = (torch.sigmoid(t_) > torch.rand_like(t_)).float().data.cpu().numpy()
        else:
            t_samples = gaussian_sampler(*torch.chunk(t_, chunks=2, dim=1)).data.cpu().numpy()
        return t_samples

    def sample_y(self, t, w=None):
        if w is None:
            w = self.sample_w()
        wt = np.concatenate([w, t], 1)
        y_ = self.mlp_y_tw(torch.from_numpy(wt).float())
        if self.outcome_distribution in ['normal', 'gaussian']:
            y_samples = gaussian_sampler(*torch.chunk(y_, chunks=2, dim=1)).data.cpu().numpy()
        elif self.outcome_distribution == 'mixed_log_logistic':
            y_samples = mixed_log_logistic_sampler(*torch.chunk(y_, chunks=3, dim=1)).data.cpu().numpy()
        elif self.outcome_distribution == 'log_logistic':
            y_samples = log_logistic_sampler(*torch.chunk(y_, chunks=2, dim=1)).data.cpu().numpy()
        else:
            raise('outcome distribution {} not implemented'.format(self.outcome_distribution))

        if self.outcome_min is not None or self.outcome_max is not None:
            return np.clip(y_samples, self.outcome_min, self.outcome_max)
        else:
            return y_samples


if __name__ == '__main__':
    from data.synthetic import generate_wty_linear_multi_w_data
    from utils import NUMPY
    from data.lalonde import load_lalonde
    import matplotlib.pyplot as plt
    import pprint
    pp = pprint.PrettyPrinter(indent=4)

    # data = generate_wty_linear_multi_w_data(500, data_format=NUMPY, wdim=5)
    #
    # dgm = DataGenModel(data)
    # data_samples = dgm.sample()

    # w, t, y = generate_wty_linear_multi_w_data(500, data_format=NUMPY, wdim=5)
    # mlp = MLP(w, t, y)
    # data_samples = mlp.sample()
    # mlp.plot_ty_dists()
    # uni_metrics = mlp.get_univariate_quant_metrics()
    # multi_ty_metrics = mlp.get_multivariate_quant_metrics(include_w=False, n_permutations=10)
    # multi_wty_metrics = mlp.get_multivariate_quant_metrics(include_w=True, n_permutations=10)

    w, t, y = load_lalonde()

    logit_pi = torch.zeros(1, requires_grad=True)
    log_alpha = torch.zeros(1, requires_grad=True)
    log_beta_ = torch.zeros(1, requires_grad=True)
    y_torch = torch.from_numpy(y/y.max()).float()
    for i in range(500):
        logit_pi.grad = None
        log_alpha.grad = None
        log_beta_.grad = None
        nll = - log_mixed_log_logistic(y_torch, logit_pi.expand(len(y)),
                                       log_alpha.expand(len(y)),
                                       log_beta_.expand(len(y))).mean()
        nll.backward()
        logit_pi.data.sub_(0.1 * logit_pi.grad.data)
        log_alpha.data.sub_(0.1 * log_alpha.grad.data)
        log_beta_.data.sub_(0.1 * log_beta_.grad.data)


    plt.hist(y / y.max(), 50, density=True, alpha=0.5, range=(0,1))
    n_ = 1000
    ll = log_mixed_log_logistic(torch.linspace(0, 1, n_),
                                logit_pi.expand(n_),
                                log_alpha.expand(n_),
                                log_beta_.expand(n_))
    plt.plot(np.linspace(0,1,n_), np.exp(ll.data.numpy()), 'x', ms=2)

    y_samples = mixed_log_logistic_sampler(logit_pi.expand(n_),
                                           log_alpha.expand(n_),
                                           log_beta_.expand(n_)).data.numpy()
    plt.hist(y_samples, 50, density=True, alpha=0.5, range=(0,1))
    plt.legend(['data', 'density', 'samples'], loc=1)

    w_mean = w.mean(0)
    w_std = w.std(0)
    y_max = y.max()

    mlp = MLP((w-w_mean)/w_std, t, y/y_max,
              training_params=TrainingParams(lr=0.001, batch_size=256, num_epochs=2000),
              mlp_params_y_tw=MLPParams(n_hidden_layers=2, dim_h=256),
              binary_treatment=True, outcome_distribution='mixed_log_logistic',
              outcome_min=0.0, outcome_max=1.0, seed=1)
    data_samples = mlp.sample()
    # mlp.plot_ty_dists()
    uni_metrics = mlp.get_univariate_quant_metrics()
    pp.pprint(uni_metrics)
    # multi_ty_metrics = mlp.get_multivariate_quant_metrics(include_w=False, n_permutations=10)
    # multi_wty_metrics = mlp.get_multivariate_quant_metrics(include_w=True, n_permutations=10)
