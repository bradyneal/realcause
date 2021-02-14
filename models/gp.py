from models.nonlinear import MLP, TrainingParams, eval_ctx
from models import preprocess
from models import distributions
from models.distributions import functional as F
import torch
import gpytorch
import numpy as np
import warnings


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

        return (f'GPParams(kernel=gpytorch.kernels.{repr(self.kernel).split("(")[0]}(), '
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


class GPBinaryClassificationModel(gpytorch.models.AbstractVariationalGP):
    def __init__(self, train_x, kernel=gpytorch.kernels.RBFKernel(),
                 var_dist=gpytorch.variational.MeanFieldVariationalDistribution):
        variational_distribution = var_dist(train_x.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(self, train_x, variational_distribution)
        # noinspection PyDeprecation
        super(GPBinaryClassificationModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred


class GPClassificationModel(gpytorch.models.AbstractVariationalGP):
    # https://arxiv.org/abs/1611.00336
    def __init__(self, train_x, kernel=gpytorch.kernels.RBFKernel(),
                 var_dist=gpytorch.variational.MeanFieldVariationalDistribution, num_tasks=2):
        print(train_x.size())
        variational_distribution = var_dist(train_x.size(0))
        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.VariationalStrategy(self, train_x, variational_distribution), num_tasks)
        # noinspection PyDeprecation
        super(GPClassificationModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.MultitaskMean(gpytorch.means.ConstantMean(), num_tasks)
        self.covar_module = gpytorch.kernels.MultitaskKernel(gpytorch.kernels.ScaleKernel(kernel), num_tasks)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred


# noinspection PyShadowingNames
class AtomicExactGPModel:

    def __init__(self, x, y, dist, num_tasks, kernel, kernel_a=None, var_dist_a=None):
        self.x = x
        self.y = y
        self.dist = dist
        self.networks = list()
        if hasattr(dist, 'atoms'):
            assert kernel_a is not None and var_dist_a is not None
            self.a_, self.ind_non_atoms = extract_atom_clf_dataset(y, dist.atoms)

            self.gp_a_x = GPClassificationModel(x, kernel=kernel_a, var_dist=var_dist_a, num_tasks=num_tasks)

            self.networks.append(self.gp_a_x)

            # atom marginal likelihood
            self.likelihood_a = gpytorch.likelihoods.SoftmaxLikelihood(
                num_features=num_tasks,
                num_classes=len(dist.atoms) + 1, mixing_weights=True)  # todo: check mixing weight
            self.mll_a = gpytorch.mlls.variational_elbo.VariationalELBO(
                self.likelihood_a, self.gp_a_x, self.a_[:, 0].numel())

            self.gp_y_x = ExactGPModel(x[self.ind_non_atoms[:, 0]], y[self.ind_non_atoms[:, 0]][:, 0], kernel=kernel)
            self.atomic = True

        else:
            # outcome network
            self.gp_y_x = ExactGPModel(x, y[:, 0], kernel)
            self.atomic = False

        self.networks.append(self.gp_y_x)
        self.likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
        self.mll_y = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood_y, self.gp_y_x)

    def loss(self, x, y):
        a, ind_non_atoms = None, None
        if self.networks[0].training:
            x = self.x
            y = self.y
            if self.atomic:
                a, ind_non_atoms = self.a_, self.ind_non_atoms
        else:
            if self.atomic:
                a, ind_non_atoms = \
                    extract_atom_clf_dataset(y, self.dist.atoms)

        if not self.atomic:
            output_y = self.gp_y_x(x)
            loss_y = - self.mll_y(output_y, y[:, 0])
        else:
            assert a is not None and ind_non_atoms is not None
            output_a = self.gp_a_x(x)
            loss_a = - self.mll_a(output_a, a[:, 0])

            output_y = self.gp_y_x(x[ind_non_atoms[:, 0]])
            loss_y = - self.mll_y(output_y, y[ind_non_atoms[:, 0]][:, 0])

            loss_y += loss_a

        return loss_y

    def sample(self, x):
        with eval_ctx(self):
            pred = self.gp_y_x(x)
            y_samples = pred.sample(sample_shape=torch.Size((1,)))[0].unsqueeze(1).data.cpu().numpy()

        if self.atomic:
            with eval_ctx(self):
                pred = self.likelihood_a(self.gp_a_x(x))
                a_ = pred.sample()[0].unsqueeze(1).float().data.cpu().numpy()

            atom_masks = list()
            for j in range(len(self.dist.atoms)):
                atom_masks.append(a_ == j)
            non_atom_mask = a_ == len(self.dist.atoms)

            for atom_mask, atom in zip(atom_masks, self.dist.atoms):
                a_[atom_mask] = atom
            a_[non_atom_mask] = y_samples[non_atom_mask]
            y_samples = a_

        return y_samples


def extract_atom_clf_dataset(x, atoms):
    """
    :param x: pytorch tensor (vector)
    :param atoms:
    :return:
    """

    ind_atoms = torch.zeros_like(x).bool()
    x_labels = torch.zeros_like(x) + len(atoms)
    for j, atom in enumerate(atoms):
        ind_ = x == atom
        ind_atoms += ind_
        x_labels[ind_] = torch.zeros_like(x_labels[ind_]) + j

    ind_non_atoms = ~ind_atoms
    return x_labels, ind_non_atoms


# noinspection PyShadowingNames
class GPModel(MLP):
    # noinspection PyAttributeOutsideInit

    # todo: kernel / var dist for atom classification (now using the same as treatment)
    def build_networks(self):
        self.GP_params_t_w = self.network_params['gp_t_w']
        self.GP_params_y_tw = self.network_params['gp_y_tw']

        # cast the transformed dataset as pytorch tensor
        self.w_transformed_ = torch.from_numpy(self.w_transformed).float()
        self.t_transformed_ = torch.from_numpy(self.t_transformed).float()
        self.y_transformed_ = torch.from_numpy(self.y_transformed).float()

        # init network list
        self.networks = list()

        # treatment network
        self.gp_t_w = GPBinaryClassificationModel(self.w_transformed_,
                                                  kernel=self.GP_params_t_w.kernel,
                                                  var_dist=self.GP_params_t_w.var_dist)
        self.networks.append(self.gp_t_w)

        # treatment marginal likelihood
        self.likelihood_t = gpytorch.likelihoods.BernoulliLikelihood()
        self.mll_t = gpytorch.mlls.variational_elbo.VariationalELBO(
            self.likelihood_t, self.gp_t_w, self.t_transformed_[:, 0].numel())

        # outcome network(s)
        # todo: tarnet GP: separate y for t=0 and t=1
        self.gp_y_tw = AtomicExactGPModel(
            x=torch.cat([self.w_transformed_, self.t_transformed_], 1),
            y=self.y_transformed_,
            dist=self.outcome_distribution,
            kernel=self.GP_params_y_tw.kernel,
            kernel_a=self.GP_params_t_w.kernel,
            var_dist_a=self.GP_params_t_w.var_dist,
            num_tasks=32 if 'num_tasks' not in self.additional_args.keys() else self.additional_args['num_tasks']
        )
        self.networks += self.gp_y_tw.networks

    def _get_loss(self, w, t, y):

        output_t = self.gp_t_w(w)
        loss_t = - self.mll_t(output_t, t[:, 0])

        loss_y = self.gp_y_tw.loss(torch.cat([w, t], 1), y)

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

        y_samples = self.gp_y_tw.sample(torch.from_numpy(wt).float())

        # todo: fix
        if ret_counterfactuals:
            return y_samples, y_samples

        if self.outcome_min is not None or self.outcome_max is not None:
            return np.clip(y_samples, self.outcome_min, self.outcome_max)
        else:
            return y_samples

    def mean_y(self, t, w):
        warnings.warn(f'mean not implemented for GP')
        return torch.zeros(*t.shape)


# noinspection PyShadowingNames
class TarGPModel(MLP):
    # noinspection PyAttributeOutsideInit

    # todo: kernel / var dist for atom classification (now using the same as treatment)
    def build_networks(self):
        self.GP_params_t_w = self.network_params['gp_t_w']
        self.GP_params_y_tw = self.network_params['gp_y_tw']

        # cast the transformed dataset as pytorch tensor
        self.w_transformed_ = torch.from_numpy(self.w_transformed).float()
        self.t_transformed_ = torch.from_numpy(self.t_transformed).float()
        self.y_transformed_ = torch.from_numpy(self.y_transformed).float()

        self.w0_transformed_ = torch.from_numpy(self.w_transformed[self.t_transformed[:, 0] == 0]).float()
        self.w1_transformed_ = torch.from_numpy(self.w_transformed[self.t_transformed[:, 0] == 1]).float()
        self.y0_transformed_ = torch.from_numpy(self.y_transformed[self.t_transformed[:, 0] == 0]).float()
        self.y1_transformed_ = torch.from_numpy(self.y_transformed[self.t_transformed[:, 0] == 1]).float()

        # init network list
        self.networks = list()

        # treatment network
        self.gp_t_w = GPBinaryClassificationModel(self.w_transformed_,
                                                  kernel=self.GP_params_t_w.kernel,
                                                  var_dist=self.GP_params_t_w.var_dist)
        self.networks.append(self.gp_t_w)

        # treatment marginal likelihood
        self.likelihood_t = gpytorch.likelihoods.BernoulliLikelihood()
        self.mll_t = gpytorch.mlls.variational_elbo.VariationalELBO(
            self.likelihood_t, self.gp_t_w, self.t_transformed_[:, 0].numel())

        # outcome network(s)
        # todo: tarnet GP: separate y for t=0 and t=1
        self.gp_y0_tw = AtomicExactGPModel(
            x=self.w0_transformed_,
            y=self.y0_transformed_,
            dist=self.outcome_distribution,
            kernel=self.GP_params_y_tw.kernel,
            kernel_a=self.GP_params_t_w.kernel,
            var_dist_a=self.GP_params_t_w.var_dist,
            num_tasks=32 if 'num_tasks' not in self.additional_args.keys() else self.additional_args['num_tasks']
        )
        self.gp_y1_tw = AtomicExactGPModel(
            x=self.w1_transformed_,
            y=self.y1_transformed_,
            dist=self.outcome_distribution,
            kernel=self.GP_params_y_tw.kernel,
            kernel_a=self.GP_params_t_w.kernel,
            var_dist_a=self.GP_params_t_w.var_dist,
            num_tasks=32 if 'num_tasks' not in self.additional_args.keys() else self.additional_args['num_tasks']
        )
        self.networks += self.gp_y0_tw.networks
        self.networks += self.gp_y1_tw.networks

    def _get_loss(self, w, t, y):
        w0 = w[t[:, 0] == 0]
        w1 = w[t[:, 0] == 1]
        y0 = y[t[:, 0] == 0]
        y1 = y[t[:, 0] == 1]

        output_t = self.gp_t_w(w)
        loss_t = - self.mll_t(output_t, t[:, 0])

        # todo: make sure batch size of gp is always the entire set
        loss_y0 = self.gp_y0_tw.loss(w0, y0)
        loss_y1 = self.gp_y1_tw.loss(w1, y1)
        loss_y = (loss_y0 * (t == 0).float().sum() + loss_y1 * (t == 1).float().sum()) / t.size(0)

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
        w = torch.from_numpy(w).float()

        y0_samples = self.gp_y0_tw.sample(w)
        y1_samples = self.gp_y1_tw.sample(w)
        if self.outcome_min is not None or self.outcome_max is not None:
            y0_samples = np.clip(y0_samples, self.outcome_min, self.outcome_max)
            y1_samples = np.clip(y1_samples, self.outcome_min, self.outcome_max)

        # todo: fix
        if ret_counterfactuals:
            return y0_samples, y1_samples
        else:
            return y0_samples * (1-t) + y1_samples * t

    def mean_y(self, t, w):
        warnings.warn(f'mean not implemented for GP')
        return torch.zeros(*t.shape)


if __name__ == "__main__":
    from data.lalonde import load_lalonde
    import matplotlib.pyplot as plt
    import pprint

    pp = pprint.PrettyPrinter(indent=4)
    TarGPModel = TarGPModel

    dataset = 1
    network_params = _DEFAULT_GP.copy()
    dist = distributions.MixedDistribution([0.0], distributions.FactorialGaussian())
    # dist = distributions.FactorialGaussian()
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

    mdl = TarGPModel(w, t, y,
                     training_params=training_params,
                     network_params=network_params,
                     binary_treatment=True, outcome_distribution=dist,
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
