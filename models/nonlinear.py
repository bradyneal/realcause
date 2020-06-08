import numpy as np
from models import distributions
from models.base import BaseGenModel, Preprocess, PlaceHolderTransform
import torch
from torch import nn
from torch.utils import data
from itertools import chain


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


class Centering(Preprocess):
    def __init__(self, data=None, mean=None):
        assert data is not None or mean is not None, 'at least one of data or mean must be provided'
        if data is not None:
            self.m = data.mean(0).astype('float32')
            if mean is not None:
                assert np.isclose(self.m, mean), 'mean of data is not close to the provided value'
        else:
            self.m = np.cast['float32'](mean)

    def transform(self, x):
        return x - self.m

    def untransform(self, x):
        return x + self.m


class Scaling(Preprocess):
    def __init__(self, s):
        self.s = np.cast['float32'](s)

    def transform(self, x):
        return x * self.s

    def untransform(self, x):
        return x / self.s


class VarianceRescaling(Scaling):
    def __init__(self, data=None, stdv=None, gain=1.0):
        assert data is not None or stdv is not None, 'at least one of data or stdv must be provided'
        if data is not None:
            s = data.std(0)
            if stdv is not None:
                assert np.isclose(s, stdv), 'stdv of data is not close to the provided value'
        else:
            s = stdv
        super(VarianceRescaling, self).__init__(gain/s)


class SequentialTransforms(Preprocess):
    def __init__(self, *transforms):
        self.transforms = transforms

    def transform(self, x):
        for t in self.transforms:
            x = t.transform(x)
        return x

    def untransform(self, x):
        for t in reversed(self.transforms):
            x = t.untransform(x)
        return x

class Standardize(Preprocess):
    def __init__(self, data):
        self.t = SequentialTransforms(
            Centering(data),
            VarianceRescaling(data)
        )

    def transform(self, x):
        return self.t.transform(x)

    def untransform(self, x):
        return self.t.untransform(x)


class CausalDataset(data.Dataset):
    def __init__(self, w, t, y, wtype='float32', ttype='float32', ytype='float32',
                 w_transform:Preprocess=PlaceHolderTransform(),
                 t_transform:Preprocess=PlaceHolderTransform(),
                 y_transform:Preprocess=PlaceHolderTransform()):
        self.w = w.astype(wtype)
        self.t = t.astype(ttype)
        self.y = y.astype(ytype)

        # todo: no need anymore, remove?
        self.w_transform = w_transform
        self.t_transform = t_transform
        self.y_transform = y_transform

    def __len__(self):
        return self.w.shape[0]

    def __getitem__(self, index):
        return self.w_transform.transform(self.w[index]), \
               self.t_transform.transform(self.t[index]), \
               self.y_transform.transform(self.y[index])



# TODO: for more complex w, we might need to share parameters (dependent on the problem)
class MLP(BaseGenModel):

    def __init__(self, w, t, y, seed=1,
                 mlp_params_t_w=MLPParams(),
                 mlp_params_y_tw=MLPParams(),
                 training_params = TrainingParams(),
                 binary_treatment=False,
                 outcome_distribution:distributions.BaseDistribution=distributions.FactorialGaussian(),
                 outcome_min=None,
                 outcome_max=None,
                 w_transform:Preprocess=PlaceHolderTransform(),
                 t_transform:Preprocess=PlaceHolderTransform(),
                 y_transform:Preprocess=PlaceHolderTransform()
                 ):
        super(MLP, self).__init__(*self._matricize((w, t, y)), seed=seed,
                                  w_transform=w_transform,
                                  t_transform=t_transform,
                                  y_transform=y_transform)

        self.binary_treatment = binary_treatment
        if binary_treatment: # todo: input?
            self.treatment_distribution = distributions.Bernoulli()
        else:
            self.treatment_distribution = distributions.FactorialGaussian()
        self.outcome_distribution = outcome_distribution
        self.outcome_min = outcome_min
        self.outcome_max = outcome_max

        self.dim_w = self.w_transformed.shape[1]
        self.dim_t = self.t_transformed.shape[1]
        self.dim_y = self.y_transformed.shape[1]

        self.MLP_params_t_w = mlp_params_t_w
        self.MLP_params_y_tw = mlp_params_y_tw
        output_multiplier_t = 1 if binary_treatment else 2
        self.mlp_t_w = self._build_mlp(self.dim_w, self.dim_t, mlp_params_t_w, output_multiplier_t)
        self.mlp_y_tw = self._build_mlp(self.dim_w+self.dim_t, self.dim_y, mlp_params_y_tw,
                                        outcome_distribution.num_params)

        # TODO: multiple optimizers ?
        self.training_params = training_params
        self.optim = training_params.optim(
            chain(self.mlp_t_w.parameters(), self.mlp_y_tw.parameters()),
            training_params.lr, **training_params.optim_args
        )

        # TODO: binary treatment -> long data type
        self.data_loader = data.DataLoader(CausalDataset(self.w_transformed, self.t_transformed, self.y_transformed),
                                           batch_size=training_params.batch_size,
                                           shuffle=True)

        # self._train()

    def _matricize(self, data):
        return [np.reshape(d, [d.shape[0], -1]) for d in data]

    def _build_mlp(self, dim_x, dim_y, MLP_params=MLPParams(), output_multiplier=2):
        dim_h = MLP_params.dim_h
        hidden_layers = [nn.Linear(dim_x, dim_h), MLP_params.activation]
        for _ in range(MLP_params.n_hidden_layers-1):
            hidden_layers += [nn.Linear(dim_h, dim_h), MLP_params.activation]
        hidden_layers += [nn.Linear(dim_h, dim_y * output_multiplier)]
        return nn.Sequential(*hidden_layers)

    def _train(self):
        c = 0
        for _ in range(self.training_params.num_epochs):
            for w, t, y in self.data_loader:
                self.optim.zero_grad()
                t_ = self.mlp_t_w(w)
                y_ = self.mlp_y_tw(torch.cat([w,t], dim=1))
                loss_t = self.treatment_distribution.loss(t, t_)
                loss_y = self.outcome_distribution.loss(y, y_)
                loss = loss_t + loss_y
                # TODO: learning rate can be separately adjusted by weighting the losses here
                loss.backward()
                self.optim.step()

                if self.training_params.verbose and c % self.training_params.print_every_iters == 0:
                    print("Iteration {}: {} {}".format(c, loss_t, loss_y))
                c += 1

    def _sample_t(self, w=None):
        t_ = self.mlp_t_w(torch.from_numpy(w).float())
        return self.treatment_distribution.sample(t_)

    def _sample_y(self, t, w=None):
        wt = np.concatenate([w, t], 1)
        y_ = self.mlp_y_tw(torch.from_numpy(wt).float())
        y_samples = self.outcome_distribution.sample(y_)

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
    # w, t, y = load_lalonde(rct=True)
    # w, t, y = load_lalonde(obs_version='cps1')

    # logit_pi = torch.zeros(1, requires_grad=True)
    # log_alpha = torch.zeros(1, requires_grad=True)
    # log_beta_ = torch.zeros(1, requires_grad=True)
    param = torch.zeros(1, 4, requires_grad=True)
    y_torch = torch.from_numpy(y/y.max()).float()[:,None]
    dist = distributions.MixedDistribution([0.0], distributions.LogLogistic())
    for i in range(500):
        param.grad = None
        nll = - dist.likelihood(y_torch, param.expand(len(y), -1)).mean()
        nll.backward()
        param.data.sub_(0.01 * param.grad.data)
        print(i)


    plt.hist(y / y.max(), 50, density=True, alpha=0.5, range=(0,1))
    n_ = 1000
    ll = dist.likelihood(torch.linspace(0, 1, n_)[:, None], param.expand(n_, -1))
    plt.plot(np.linspace(0,1,n_), np.exp(ll.data.numpy()), 'x', ms=2)

    y_samples = dist.sample(param.expand(n_, -1))
    plt.hist(y_samples, 50, density=True, alpha=0.5, range=(0,1))
    plt.legend(['data', 'density', 'samples'], loc=1)

    mlp = MLP(w, t, y,
              training_params=TrainingParams(lr=0.0005, batch_size=128, num_epochs=500),
              mlp_params_y_tw=MLPParams(n_hidden_layers=2, dim_h=256),
              binary_treatment=True, outcome_distribution=dist,
              outcome_min=0.0, outcome_max=1.0, seed=1,
              w_transform=Standardize(w), y_transform=Scaling(1/y.max()))
    mlp._train()
    data_samples = mlp.sample()
    # mlp.plot_ty_dists()
    uni_metrics = mlp.get_univariate_quant_metrics()
    pp.pprint(uni_metrics)
    # multi_ty_metrics = mlp.get_multivariate_quant_metrics(include_w=False, n_permutations=10)
    multi_wty_metrics = mlp.get_multivariate_quant_metrics(include_w=True, n_permutations=10)
