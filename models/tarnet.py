from models.nonlinear import MLP, MLPParams, TrainingParams
from models import preprocess
from models import distributions
import torch
import numpy as np


_DEFAULT_TARNET = dict(
    mlp_params_w=MLPParams(),
    mlp_params_t_w=MLPParams(),
    mlp_params_y0_w=MLPParams(),
    mlp_params_y1_w=MLPParams(),
)


class TarNet(MLP):
    # noinspection PyAttributeOutsideInit
    def build_networks(self):
        self.MLP_params_w = self.network_params['mlp_params_w']
        self.MLP_params_t_w = self.network_params['mlp_params_t_w']
        self.MLP_params_y0_w = self.network_params['mlp_params_y0_w']
        self.MLP_params_y1_w = self.network_params['mlp_params_y1_w']

        output_multiplier_t = 1 if self.binary_treatment else 2
        self._mlp_w = self._build_mlp(self.dim_w, self.MLP_params_w.dim_h, self.MLP_params_w, 1)
        self._mlp_t_w = self._build_mlp(self.MLP_params_w.dim_h, self.dim_t, self.MLP_params_t_w, output_multiplier_t)
        self._mlp_y0_w = self._build_mlp(self.MLP_params_w.dim_h, self.dim_y, self.MLP_params_y0_w,
                                         self.outcome_distribution.num_params)
        self._mlp_y1_w = self._build_mlp(self.MLP_params_w.dim_h, self.dim_y, self.MLP_params_y1_w,
                                         self.outcome_distribution.num_params)
        self.networks = [self._mlp_w, self._mlp_t_w, self._mlp_y0_w, self._mlp_y1_w]

    def mlp_w(self, w):
        return self.MLP_params_w.activation(self._mlp_w(w))

    def mlp_t_w(self, w):
        return self._mlp_t_w(self.mlp_w(w))

    def mlp_y_tw(self, wt, ret_counterfactuals=False):
        """
        :param wt: concatenation of w and t
        :return: parameter of the conditional distribution p(y|t,w)
        """
        w, t = wt[:, :-1], wt[:, -1:]
        w = self.mlp_w(w)
        y0 = self._mlp_y0_w(w)
        y1 = self._mlp_y1_w(w)
        if ret_counterfactuals:
            return y0, y1
        else:
            return y0 * (1 - t) + y1 * t


    def _get_loss(self, w, t, y):
        # compute w_ only once
        w_ = self.mlp_w(w)
        t_ = self._mlp_t_w(w_)
        if self.ignore_w:
            w_ = torch.zeros_like(w_)

        y0 = self._mlp_y0_w(w_)
        y1 = self._mlp_y1_w(w_)
        y_ = y0 * (1 - t) + y1 * t

        loss_t = self.treatment_distribution.loss(t, t_)
        loss_y = self.outcome_distribution.loss(y, y_)
        loss = loss_t + loss_y
        return loss, loss_t, loss_y


if __name__ == "__main__":
    from data.lalonde import load_lalonde
    import matplotlib.pyplot as plt
    import pprint

    pp = pprint.PrettyPrinter(indent=4)

    dataset = 1
    network_params = _DEFAULT_TARNET.copy()
    if dataset == 1:
        w, t, y = load_lalonde()
        dist = distributions.MixedDistribution([0.0], distributions.LogLogistic())
        training_params = TrainingParams(lr=0.0005, batch_size=128, num_epochs=100, verbose=False)
        early_stop = True
        ignore_w = False
    elif dataset == 2:
        w, t, y = load_lalonde(rct=True)
        # dist = distributions.MixedDistribution([0.0], distributions.LogLogistic())
        dist = distributions.FactorialGaussian()
        training_params = TrainingParams(lr=0.001, batch_size=64, num_epochs=200)
        early_stop = True
        ignore_w = False
    elif dataset == 3:
        w, t, y = load_lalonde(obs_version="cps1")
        dist = distributions.MixedDistribution(
            [0.0, 25564.669921875], distributions.SigmoidFlow(ndim=10)
            # [0.0], distributions.SigmoidFlow(ndim=10)
        )
        training_params = TrainingParams(lr=0.0005, batch_size=128, num_epochs=100)
        early_stop = True
        ignore_w = False
    else:
        raise (Exception("dataset {} not implemented".format(dataset)))

    param = torch.zeros(1, dist.num_params, requires_grad=True)
    y_torch = torch.from_numpy(y / y.max()).float()[:, None]
    for i in range(100):
        param.grad = None
        nll = -dist.likelihood(y_torch, param.expand(len(y), -1)).mean()
        nll.backward()
        param.data.sub_(0.01 * param.grad.data)
        print(i)

    plt.hist(y / y.max(), 50, density=True, alpha=0.5, range=(0, 1))
    n_ = 1000
    ll = dist.likelihood(torch.linspace(0, 1, n_)[:, None], param.expand(n_, -1))
    plt.plot(np.linspace(0, 1, n_), np.exp(ll.data.numpy()), "x", ms=2)

    y_samples = dist.sample(param.expand(n_, -1))
    plt.hist(y_samples, 50, density=True, alpha=0.5, range=(0, 1))
    plt.legend(["data", "density", "samples"], loc=1)

    mdl = TarNet(w, t, y,
                 training_params=training_params,
                 network_params=network_params,
                 binary_treatment=True, outcome_distribution=dist,
                 outcome_min=0.0, outcome_max=1.0,
                 train_prop=0.5,
                 val_prop=0.1,
                 test_prop=0.4,
                 seed=1,
                 early_stop=early_stop,
                 ignore_w=ignore_w,
                 w_transform=preprocess.Standardize, y_transform=preprocess.Normalize)
    mdl.train()
    data_samples = mdl.sample()
    # mlp.plot_ty_dists()
    uni_metrics = mdl.get_univariate_quant_metrics(dataset="test")
    pp.pprint(uni_metrics)
    print("noisy ate:", mdl.noisy_ate())
