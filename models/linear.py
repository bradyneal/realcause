import numpy as np
from models.base import BaseGenModel


class LinearGenModel(BaseGenModel):

    def __init__(self, w, t, y, lambda0_t_w=1e-5, lambda0_y_tw=1e-5,
                 binary_treatment=False, **kwargs):
        super(LinearGenModel, self).__init__(*self._matricize((w, t, y)), **kwargs)
        self._train(lambda0_t_w, lambda0_y_tw)
        self.binary_treatment = binary_treatment

    def _matricize(self, data):
        return [np.reshape(d, [d.shape[0], -1]) for d in data]

    def _train(self, lambda0_t_w=1e-5, lambda0_y_tw=1e-5):
        # fitting a linear Gaussian model defined as
        # p(y | x) = N(y; beta*x, sigma^2)
        # with a prior p(beta) = N(0, sigma^2/lambda0) which amounts to L2 penalty on beta: lambda||beta||^2
        # where x and y are input and output respectively
        # TODO: learning different sigma for each output feature

        w, t, y = self.w_transformed, self.t_transformed, self.y_transformed

        # t | w
        self.beta_t_w, self.sigma_t_w = self.linear_gaussian_solver(w, t, lambda0_t_w)

        # y | t and w
        self.beta_y_tw, self.sigma_y_tw = self.linear_gaussian_solver(np.concatenate([w, t], 1), y, lambda0_y_tw)

    def _pad_with_ones(self, X):
        return np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

    def _linear_gaussian_solver(self, X, Y, lambda0=1e-5):
        beta = np.linalg.inv(X.T.dot(X) + lambda0 * np.identity(X.shape[1])).dot(X.T.dot(Y))
        var = ((Y - X.dot(beta)) ** 2).mean()
        # assuming sharing the save variance parameter across different yj's
        return beta, np.sqrt(var)

    def linear_gaussian_solver(self, X, Y, lambda0=1e-5):
        return self._linear_gaussian_solver(self._pad_with_ones(X), Y, lambda0)

    def gaussian_sampler(self, mean, sigma):
        return np.random.randn(*mean.shape) * sigma + mean

    def linear_gaussian_sampler(self, X, beta, sigma):
        mean = self._pad_with_ones(X).dot(beta)
        return self.gaussian_sampler(mean, sigma)

    def _sample_t(self, w=None):
        t_samples = self.linear_gaussian_sampler(
            w,
            self.beta_t_w,
            self.sigma_t_w
        )
        if self.binary_treatment:
            t_samples = t_samples > .5
        return t_samples

    def _sample_y(self, t, w=None):
        y_samples = self.linear_gaussian_sampler(
            np.concatenate([w, t], 1),
            self.beta_y_tw,
            self.sigma_y_tw
        )
        return y_samples

    def mean_y(self, t, w):
        X = np.concatenate([w, t], 1)
        return self._pad_with_ones(X).dot(self.beta_y_tw)


if __name__ == '__main__':
    from data.synthetic import generate_wty_linear_multi_w_data
    from utils import NUMPY

    # data = generate_wty_linear_multi_w_data(500, data_format=NUMPY, wdim=5)
    #
    # dgm = DataGenModel(data)
    # data_samples = dgm.sample()

    w, t, y = generate_wty_linear_multi_w_data(500, data_format=NUMPY, wdim=5)

    lgm = LinearGenModel(w, t, y)
    data_samples = lgm.sample()
    lgm.plot_ty_dists()
    uni_metrics = lgm.get_univariate_quant_metrics()
    multi_ty_metrics = lgm.get_multivariate_quant_metrics(include_w=False)
    multi_wty_metrics = lgm.get_multivariate_quant_metrics(include_w=True)
