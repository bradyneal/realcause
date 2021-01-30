import warnings
import numpy as np
import torch
from torch.nn import functional as F
from models.distributions.flows import sigmoid_flow, sigmoid_flow_inverse

Log2PI = float(np.log(2 * np.pi))
VALID_BASE = ['uniform', 'normal', 'gaussian']


def log_normal(x, mean, log_var, eps=0.00001):
    z = - 0.5 * Log2PI
    return - (x - mean) ** 2 / (2. * torch.exp(log_var) + eps) - log_var / 2. + z


def log_standard_normal(x):
    return log_normal(x, torch.zeros_like(x), torch.zeros_like(x), eps=0.0)


def log_log_normal(x, mean, log_var, eps=0.00001):
    return log_normal(torch.log(x), mean, log_var, eps) - torch.log(x)


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


def log_exponential(x, log_lambda):
    return log_lambda - torch.exp(log_lambda) * x


def log_bernoulli(x, logit):
    return - F.binary_cross_entropy_with_logits(logit, x, reduction='none').sum(-1)


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


def bernoulli_sampler(logit, overlap=1):
    """
    Sample (treatment vector) from bernoulli.

    :param logit: p = sigmoid(logit)
    :param overlap: if 1, leave treatment untouched;
        if 0, push p(T = 1 | w) to 0 for all w where p(T = 1 | w) < 0.5 and
        and push p(T = 1 | w) to 1 for all w where p(T = 1 | w) >= 0.5
        if 0 < overlap < 1, do a linear interpolation of the above
    :return: sampled treatment
    """
    p = torch.sigmoid(logit).float().data.cpu().numpy()

    assert 0 <= overlap <= 1
    if overlap < 1:
        likely_treated = p >= 0.5
        likely_control = np.logical_not(likely_treated)
        p = likely_treated * (overlap * p + (1 - overlap) * 1) + \
            likely_control * (overlap * p + (1 - overlap) * 0)

    t = p > np.random.rand(*p.shape)

    return t.astype('float32')


def exponential_sampler(log_lambda):
    # F(x) = 1 - exp(-lambda * x)
    # F^-1(u) = - log(1-u) / lambda
    return - torch.log(1 - torch.rand_like(log_lambda) + 1e-8) / (torch.exp(log_lambda) + 1e-8)


class BaseDistribution(object):
    """Distribution with batchified likelihood function and sampling function"""

    dists = dict()
    dist_names = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.dists[cls.__name__] = cls
        cls.dist_names.append(cls.__name__)

    def likelihood(self, x, params):
        raise NotImplementedError

    def sample(self, params):
        raise NotImplementedError

    def loss(self, x, params):
        return - self.likelihood(x, params).mean()

    @property
    def num_params(self):
        raise NotImplementedError

    def mean(self, params):
        raise NotImplementedError


class Bernoulli(BaseDistribution):
    def likelihood(self, x, params):
        return log_bernoulli(x, params)

    def sample(self, params, overlap=1):
        return bernoulli_sampler(params)

    @property
    def num_params(self):
        return 1

    def mean(self, params):
        return torch.sigmoid(params)


class Exponential(BaseDistribution):
    def likelihood(self, x, params):
        return log_exponential(x, params)

    def sample(self, params):
        return exponential_sampler(params)

    @property
    def num_params(self):
        return 1

    def mean(self, params):
        return 1 / (torch.exp(params) + 1e-8)


class FactorialGaussian(BaseDistribution):
    def likelihood(self, x, params):
        return log_normal(x, *torch.chunk(params, chunks=2, dim=1)).sum(-1)

    def sample(self, params):
        return gaussian_sampler(*torch.chunk(params, chunks=2, dim=1)).data.cpu().numpy()

    @property
    def num_params(self):
        return 2

    def mean(self, params):
        return torch.chunk(params, chunks=2, dim=1)[0]


# aliasing
Normal = FactorialGaussian
Gaussian = FactorialGaussian


class LogLogistic(BaseDistribution):
    def likelihood(self, x, params):
        return log_log_logistic(x, *torch.chunk(params, chunks=2, dim=1)).sum(-1)

    def sample(self, params):
        return log_logistic_sampler(*torch.chunk(params, chunks=2, dim=1)).data.cpu().numpy()

    @property
    def num_params(self):
        return 2

    def mean(self, params):
        log_alpha, log_beta_ = torch.chunk(params, chunks=2, dim=1)
        log_beta = F.softplus(log_beta_)  # log(beta) > 0 => beta > 1
        beta = torch.exp(log_beta)
        alpha = torch.exp(log_alpha)
        pi_beta = np.pi / beta
        return alpha * pi_beta / torch.sin(pi_beta)


class LogNormal(BaseDistribution):
    def likelihood(self, x, params):
        return log_log_normal(x, *torch.chunk(params, chunks=2, dim=1)).sum(-1)

    def sample(self, params):
        return torch.exp(gaussian_sampler(*torch.chunk(params, chunks=2, dim=1))).data.cpu().numpy()

    @property
    def num_params(self):
        return 2

    def mean(self, params):
        mean, log_var = torch.chunk(params, chunks=2, dim=1)
        var = torch.exp(log_var)
        return torch.exp(mean + var / 2)


class SigmoidFlow(BaseDistribution):
    def __init__(self, ndim=4, base_distribution='uniform'):
        super(SigmoidFlow, self).__init__()
        assert base_distribution in VALID_BASE, f'base_distribution not one of {VALID_BASE}, got {base_distribution}'
        self.ndim = ndim
        self.base_distribution = base_distribution
        self.logit_end = False if base_distribution == 'uniform' else True

    def forward_transform(self, x, params, logdet=0):
        y, logdet = sigmoid_flow(
            x, logdet, ndim=self.ndim, params=params.view(*x.size(), self.ndim*3), logit_end=self.logit_end)
        return y, logdet

    def likelihood(self, x, params, logdet=0):
        y, logdet = self.forward_transform(x, params, logdet)
        if self.logit_end:
            return log_standard_normal(y).sum(-1) + logdet
        else:
            return logdet

    def sample(self, params):
        n, params_dim = params.shape
        p = int(params_dim / (self.ndim * 3))
        slc = torch.chunk(params, self.ndim*3, 1)[0]
        y = torch.randn_like(slc) if self.logit_end else torch.rand_like(slc)
        x = sigmoid_flow_inverse(
            y, ndim=self.ndim, params=params.view(n, p, self.ndim*3),
            logit_end=self.logit_end, x=None, tol=1e-2, max_iter=100, lr=0.1)
        return x.data.cpu().numpy()

    @property
    def num_params(self):
        return 3 * self.ndim

    def mean(self, params):
        warnings.warn(f'mean not implemented for {self.__str__()}')
        return torch.zeros_like(params[:, :1])

    def __str__(self):
        return f'{super(SigmoidFlow, self).__str__()} ndim:{self.ndim} base_distribution:{self.base_distribution}'


class MixedDistribution(BaseDistribution):
    def __init__(self, atoms: list, dist: BaseDistribution):
        """
        :param atoms: position of the point mass
        :param dist: continuous distribution
        """
        self.num_atoms = len(atoms)
        self.atoms = atoms
        self.dist = dist

    def likelihood(self, x, params):
        # todo: slicer is so that it can potentially handle n-d array.
        #  but it doesn't handle multiple features within an axis yet

        slc = [slice(None)] * len(params.shape)
        slc[-1] = slice(0, self.num_atoms + 1)
        logit_pi = params[slc]
        log_pi = F.log_softmax(logit_pi, dim=-1)

        log_pi = log_pi.view(-1, self.num_atoms + 1)
        x_shape = x.shape
        x = x.view(-1)

        ll = torch.zeros_like(x)
        ind_atoms = torch.zeros_like(x).bool()
        for j, atom in enumerate(self.atoms):
            ind_ = x == atom
            ll[ind_] = log_pi[ind_][:, j]
            ind_atoms += ind_

        ind_non_atoms = ~ind_atoms
        x_continuous = x[ind_non_atoms]
        ll[ind_non_atoms] = log_pi[ind_non_atoms][:, -1] + \
            self.dist.likelihood(x_continuous[:, None],
                                 params[:, self.num_atoms + 1:][ind_non_atoms].view(-1, self.dist.num_params))
        return ll.view(*x_shape).sum(-1)

    def sample(self, params):
        p_ = (params.size(1) - (self.num_atoms + 1)) / self.dist.num_params
        p = int(p_)
        assert p == p_, 'dimensionality of params is wrong'

        logit_pi = params[:, :self.num_atoms + 1]
        pi = F.softmax(logit_pi, dim=-1)
        x = torch.zeros(params.size(0), p).float().to(params.device)
        u = torch.rand(params.size(0), p).to(params.device)

        cum_pi = pi.cumsum(-1)
        ind_ = u < cum_pi[:, 0:1]
        x[ind_] = self.atoms[0]

        for j in range(self.num_atoms-1):
            atom = self.atoms[j+1]
            ind_ = u >= cum_pi[:, j:j+1]
            x[ind_] = atom

        ind_non_atoms = (u >= cum_pi[:, -2:-1]).squeeze()
        x[ind_non_atoms] = torch.from_numpy(
            self.dist.sample(params[:, self.num_atoms + 1:][ind_non_atoms])).to(x.device)
        return x.to(x.device).data.cpu().numpy()

    @property
    def num_params(self):
        # num_atoms + 1 classes (last one being continuous)
        return self.num_atoms + 1 + self.dist.num_params

    def mean(self, params):
        slc = [slice(None)] * len(params.shape)
        slc[-1] = slice(0, self.num_atoms + 1)
        logit_pi = params[slc]
        pi = F.softmax(logit_pi, dim=-1)
        mean = 0
        for j, atom in enumerate(self.atoms):
            mean += pi[:, j] * atom

        return mean + pi[:, -1] * self.dist.mean(params[:, self.num_atoms + 1:])[:, 0]  # todo: keep dim?


def quick_test():
    import matplotlib.pyplot as plt
    dist = MixedDistribution([0, 1.5], FactorialGaussian())
    dist.likelihood(torch.randn(1000, 1), torch.randn(1000, 5))
    dist.sample(torch.randn(100, 5))
    plt.hist(dist.sample(torch.randn(1000, 5)), 20)


def quick_test_mean():
    n = 1000000
    for dist in [FactorialGaussian(), LogNormal(), LogLogistic(), MixedDistribution([1.0, 5.5], FactorialGaussian())]:
        num_params = dist.num_params
        params = torch.randn(10, num_params)
        mean = dist.mean(params)[:, 0]
        mean_ = dist.sample(params[None].expand(n, 10, -1).reshape(-1, dist.num_params)).reshape(n, -1).mean(0)
        print(mean - mean_)
