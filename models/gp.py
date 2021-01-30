from models.nonlinear import MLP, TrainingParams, eval_ctx
from models import preprocess
from models.distributions import functional as F
import torch
import gpytorch
import numpy as np


class GPParams:
    def __init__(self,
                 kernel=gpytorch.kernels.RBFKernel(),
                 var_dist=gpytorch.variational.MeanFieldVariationalDistribution):
        self.kernel = kernel
        self.var_dist = var_dist

    def __repr__(self):
        if self.var_dist is None:
            var_dist_name = 'None'
        else:
            var_dist_name = f'gpytorch.variational.{self.var_dist.__name__}'

        return (f'GPParams(kernel=gpytorch.kernels.{ repr(self.kernel).split("(")[0] }(), '
                f'var_dist={var_dist_name})')


_DEFAULT_GP = dict(
    gp_t_w=GPParams(),
    gp_y_tw=GPParams(),
)


class ExactGPModel(gpytorch.models.ExactGP):
    """
    GP model for regression
    taken from https://docs.gpytorch.ai/en/stable/examples/01_Exact_GPs/Simple_GP_Regression.html
    """
    def __init__(self, train_x, train_y, kernel=gpytorch.kernels.RBFKernel()):
        # note that x is (batch size, # features) and y is just (batch size, )
        super(ExactGPModel, self).__init__(train_x, train_y, gpytorch.likelihoods.GaussianLikelihood())
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPClassificationModel(gpytorch.models.AbstractVariationalGP):
    def __init__(self, train_x, kernel=gpytorch.kernels.RBFKernel(),
                 var_dist=gpytorch.variational.MeanFieldVariationalDistribution):
        variational_distribution = var_dist(train_x.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(self, train_x, variational_distribution)
        # noinspection PyDeprecation
        super(GPClassificationModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred


class GPModel(MLP):
    # noinspection PyAttributeOutsideInit
    def build_networks(self):
        self.GP_params_t_w = self.network_params['gp_t_w']
        self.GP_params_y_tw = self.network_params['gp_y_tw']

        self.w_transformed_ = torch.from_numpy(self.w_transformed).float()
        self.t_transformed_ = torch.from_numpy(self.t_transformed).float()
        self.y_transformed_ = torch.from_numpy(self.y_transformed).float()

        self.gp_t_w = GPClassificationModel(self.w_transformed_,
                                            kernel=self.GP_params_t_w.kernel, var_dist=self.GP_params_t_w.var_dist)
        self.gp_y_tw = ExactGPModel(torch.cat([self.w_transformed_, self.t_transformed_], 1), self.y_transformed_[:, 0],
                                    kernel=self.GP_params_y_tw.kernel)

        self.networks = [self.gp_t_w, self.gp_y_tw]

        # marginal likelihood functions
        self.likelihood_t = gpytorch.likelihoods.BernoulliLikelihood()
        self.mll_t = gpytorch.mlls.variational_elbo.VariationalELBO(
            self.likelihood_t, self.gp_t_w, self.t_transformed_[:, 0].numel())

        self.likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
        self.mll_y = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.likelihood_y, self.gp_y_tw)

    def _get_loss(self, w, t, y):

        if self.networks[0].training:
            w = self.w_transformed_
            t = self.t_transformed_
            y = self.y_transformed_


        output_t = self.gp_t_w(w)
        output_y = self.gp_y_tw(torch.cat([w, t], 1))

        loss_t = - self.mll_t(output_t, t[:, 0])
        loss_y = - self.mll_y(output_y, y[:, 0])
        loss = loss_t + loss_y

        return loss, loss_t, loss_y

    def _sample_t(self, w=None, overlap=0):
        with eval_ctx(self):
            pred = self.likelihood_t(self.gp_t_w(torch.from_numpy(w).float()))
            t_ = F.logit(pred.mean.unsqueeze(1))
        return self.treatment_distribution.sample(t_, overlap)

    def _sample_y(self, t, w=None, ret_counterfactuals=False):
        # todo: conditionally independent y?
        # todo: ret_counterfactuals
        if self.ignore_w:
            w = np.zeros_like(w)
        wt = np.concatenate([w, t], 1)

        with eval_ctx(self):
            pred = self.gp_y_tw(torch.from_numpy(wt).float())
            y_samples = pred.sample(sample_shape=torch.Size((1,)))[0].unsqueeze(1).data.cpu().numpy()


        if self.outcome_min is not None or self.outcome_max is not None:
            return np.clip(y_samples, self.outcome_min, self.outcome_max)
        else:
            return y_samples

    def mean_y(self, t, w):
        if self.ignore_w:
            w = np.zeros_like(w)
        wt = np.concatenate([w, t], 1)
        with eval_ctx(self):
            pred = self.gp_y_tw(torch.from_numpy(wt).float())
            mean = pred.mean.unsqueeze(1)
        return mean


if __name__ == "__main__":
    from data.lalonde import load_lalonde
    import matplotlib.pyplot as plt
    import pprint

    pp = pprint.PrettyPrinter(indent=4)

    dataset = 1
    network_params = _DEFAULT_GP.copy()
    if dataset == 1:
        w, t, y = load_lalonde()
        training_params = TrainingParams(lr=0.005, batch_size=128, num_epochs=200, verbose=False)
        early_stop = True
        ignore_w = False
    elif dataset == 2:
        w, t, y = load_lalonde(rct=True)
        training_params = TrainingParams(lr=0.001, batch_size=64, num_epochs=200)
        early_stop = True
        ignore_w = False
    elif dataset == 3:
        w, t, y = load_lalonde(obs_version="cps1")
        training_params = TrainingParams(lr=0.0005, batch_size=128, num_epochs=100)
        early_stop = True
        ignore_w = False
    else:
        raise (Exception("dataset {} not implemented".format(dataset)))

    # todo: hacked for quick test.. remove
    w, t, y = w[::5], t[::5], y[::5]
    # w, t, y = w[:400], t[:400], y[:400]

    mdl = GPModel(w, t, y,
                 training_params=training_params,
                 network_params=network_params,
                 binary_treatment=True,
                 # outcome_min=0.0, outcome_max=1.0,
                 train_prop=0.5,
                 val_prop=0.1,
                 test_prop=0.4,
                 seed=1,
                 early_stop=early_stop,
                 ignore_w=ignore_w,
                 w_transform=preprocess.Standardize, y_transform=preprocess.Standardize)
    mdl.train()
    data_samples = mdl.sample()
    # mlp.plot_ty_dists()
    uni_metrics = mdl.get_univariate_quant_metrics(dataset="test")
    pp.pprint(uni_metrics)
    print("noisy ate:", mdl.noisy_ate())

    plt.figure()
    plt.hist(mdl.y, 50, density=True, alpha=0.5)
    plt.hist(data_samples[2], 50, density=True, alpha=0.5)
    plt.legend(['gt', 'model'])
    plt.savefig('temp.png')
