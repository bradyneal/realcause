from abc import ABCMeta, abstractmethod
from numbers import Number
import numpy as np
import torch
from scipy import stats

from plotting import compare_joints, compare_bivariate_marginals
from utils import T, Y, to_np_vectors, to_torch_variable, permutation_test

MODEL_LABEL = 'model'
TRUE_LABEL = 'true'
T_MODEL_LABEL = '{} ({})'.format(T, MODEL_LABEL)
Y_MODEL_LABEL = '{} ({})'.format(Y, MODEL_LABEL)
T_TRUE_LABEL = '{} ({})'.format(T, TRUE_LABEL)
Y_TRUE_LABEL = '{} ({})'.format(Y, TRUE_LABEL)


class BaseGenModelMeta(ABCMeta):
    """
    Forces subclasses to implement abstract_attributes
    """
    required_attributes = []

    def __call__(cls, *args, **kwargs):
        obj = super(BaseGenModelMeta, cls).__call__(*args, **kwargs)
        missing_attributes = []
        for attr_name in obj.abstract_attributes:
            if not hasattr(obj, attr_name):
                missing_attributes.append(attr_name)
        if len(missing_attributes) == 1:
            raise TypeError("Can't instantiate abstract class {} with abstract attribute '{}'. "
                            "You must set self.{} in the constructor."
                            .format(cls.__name__, missing_attributes[0], missing_attributes[0]))
        elif len(missing_attributes) > 1:
            raise TypeError("Can't instantiate abstract class {} with abstract attributes {}. "
                            "For example, you must set self.{} in the constructor."
                            .format(cls.__name__, missing_attributes, missing_attributes[0]))

        return obj


class BaseGenModel(object, metaclass=BaseGenModelMeta):
    """
    Abstract class for generative models. Implementations of 2 methods and
    3 attributes are required.

    2 methods:
        sample_t(w) - models p(t | w)
        sample_y(t, w) - models p(y | t, w)

    3 attributes:
        w - covariates from real data
        t - treatments from real data
        y - outcomes from real data
    """

    abstract_attributes = ['w', 't', 'y']

    def __init__(self, w, t, y):
        self.w = w
        self.t = t
        self.y = y

    def sample_w(self):
        return self.w

    @abstractmethod
    def sample_t(self, w):
        pass

    @abstractmethod
    def sample_y(self, t, w):
        pass

    # @abstractmethod
    # def set_seed(self, seed):
    #     pass

    def sample(self, seed=0):
        # self.set_seed(seed)
        w = self.sample_w()
        t = self.sample_t(w)
        y = self.sample_y(t, w)
        return w, t, y

    def sample_interventional(self, t, w=None):
        if w is None:
            w = self.sample_w()
        if not isinstance(w, np.ndarray):
            raise ValueError('Unsupported data type: {} ... only numpy is currently supported'.format(type(w)))
        if isinstance(t, Number):
            t = np.full_like(self.t, t)
        return self.sample_y(t, w)

    def interventional_mean(self, t, w=None):
        samples = self.sample_interventional(t=t, w=w)
        return samples.mean()

    def ate(self, t1=1, t0=0, w=None):
        return self.interventional_mean(t=t1, w=w) - self.interventional_mean(t=t0, w=w)

    def ite(self, t1=1, t0=0, w=None):
        if w is None:
            w = self.sample_w()
        return self.sample_interventional(t=t1, w=w) - self.sample_interventional(t=t0, w=w)

    def plot_ty_dists(self, joint=True, marginal_hist=True, marginal_qq=True, title=True, name=None,
                      file_ext='pdf', thin_model=None, thin_true=None, joint_kwargs={}, test=False):
        """
        Creates up to 3 different plots of the real data and the corresponding model

        :param joint: boolean for whether to plot p(t, y)
        :param marginal_hist: boolean for whether to plot the p(t) and p(y) histograms
        :param marginal_qq: boolean for whether to plot the p(t) and p(y) Q-Q plots
        :param name: name to use in plot titles and saved files defaults to name of class
        :param file_ext: file extension to for saving plots (e.g. 'pdf', 'png', etc.)
        :param thin_model: thinning interval for the model data
        :param thin_true: thinning interval for the real data
        :param joint_kwargs: kwargs passed to sns.kdeplot() for p(t, y)
        :param test: if True, does not show or save plots
        :return:
        """
        if name is None:
            name = self.__class__.__name__

        _, t_model, y_model = to_np_vectors(self.sample(), thin_interval=thin_model)
        t_true, y_true = to_np_vectors((self.t, self.y), thin_interval=thin_true)

        if joint:
            compare_joints(t_model, y_model, t_true, y_true,
                           xlabel1=T_MODEL_LABEL, ylabel1=Y_MODEL_LABEL,
                           xlabel2=T_TRUE_LABEL, ylabel2=Y_TRUE_LABEL,
                           xlabel=T, ylabel=Y,
                           label1=MODEL_LABEL, label2=TRUE_LABEL,
                           save_fname='{}_ty_joints.{}'.format(name, file_ext),
                           title=title, name=name, test=test, kwargs=joint_kwargs)

        if marginal_hist or marginal_qq:
            compare_bivariate_marginals(t_true, t_model, y_true, y_model,
                                        xlabel=T, ylabel=Y,
                                        label1=TRUE_LABEL, label2=MODEL_LABEL,
                                        hist=marginal_hist, qqplot=marginal_qq,
                                        save_hist_fname='{}_ty_marginal_hists.{}'.format(name, file_ext),
                                        save_qq_fname='{}_ty_marginal_qqplots.{}'.format(name, file_ext),
                                        title=title, name=name, test=test)

    def get_univariate_quant_metrics(self, thin_model=None, thin_true=None):
        """
        Calculates quantitative metrics for the difference between p(t) and
        p_model(t) and the difference between p(y) and p_model(y)

        :param thin_model: thinning interval for the model data
        :param thin_true: thinning interval for the real data
        :return: {
            't_ks_pval': ks p-value with null that t_model and t_true are from the same distribution
            'y_ks_pval': ks p-value with null that y_model and y_true are from the same distribution
            't_wasserstein1_dist': wasserstein1 distance between t_true and t_model
            'y_wasserstein1_dist': wasserstein1 distance between y_true and y_model
        }
        """
        _, t_model, y_model = to_np_vectors(self.sample(), thin_interval=thin_model)
        t_true, y_true = to_np_vectors((self.t, self.y), thin_interval=thin_true)
        ks_label = '_ks_pval'
        wasserstein_label = '_wasserstein1_dist'
        metrics = {
            T + ks_label: stats.ks_2samp(t_model, t_true).pvalue,
            Y + ks_label: stats.ks_2samp(y_model, y_true).pvalue,
            T + wasserstein_label: stats.wasserstein_distance(t_model, t_true),
            Y + wasserstein_label: stats.wasserstein_distance(y_model, y_true),
        }
        return metrics

    def get_multivariate_quant_metrics(self, include_w=True, norm=2, k=1,
                                       alphas=None, n_permutations=1000):
        """
        Computes Wasserstein-1 and Wasserstein-2 distances. Also computes all the
        test statistics and p-values for the multivariate two sample tests from
        the torch_two_sample package. See that documentation for more info on
        the specific tests: https://torch-two-sample.readthedocs.io/en/latest/

        :param include_w: If False, test if p(t, y) = p_model(t, y).
        If True, test if p(w, t, y) = p(w, t, y).
        :param norm: norm used for Friedman-Rafsky test and kNN test
        :param k: number of nearest neighbors to use for kNN test
        :param alphas: list of kernel parameters for MMD test
        :param n_permutations: number of permutations for each test
        :return: {
            'wasserstein1_dist': wasserstein1 distance between p_true and p_model
            'wasserstein2_dist': wasserstein2 distance between p_true and p_model
            'Friedman-Rafsky pval': p-value for Friedman-Rafsky test with null
                that p_true and p_model are from the same distribution
            'kNN pval': p-value for kNN test with null that p_true and p_model are from the same distribution
            'MMD pval': p-value for MMD test with null that p_true and p_model are from the same distribution
            'Energy pval': p-value for the energy test with null that p_true and p_model are from the same distribution
        }
        """
        try:
            import ot
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(str(e) + ' ... Install: conda install cython && conda install -c conda-forge pot')
        try:
            from torch_two_sample.statistics_nondiff import FRStatistic, KNNStatistic
            from torch_two_sample.statistics_diff import MMDStatistic, EnergyStatistic
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(str(e) + ' ... Install: pip install git+git://github.com/josipd/torch-two-sample')

        t_model, y_model = to_np_vectors(self.sample()[1:], column_vector=True)
        t_true, y_true = to_np_vectors((self.t, self.y), column_vector=True)
        model_samples = np.hstack((t_model, y_model))
        true_samples = np.hstack((t_true, y_true))

        if include_w:
            model_samples = np.hstack((self.w, model_samples))
            true_samples = np.hstack((self.w, true_samples))

        assert model_samples.shape[0] == true_samples.shape[0]
        n = model_samples.shape[0]

        a, b = np.ones((n,)) / n, np.ones((n,)) / n  # uniform distribution on samples

        def calculate_wasserstein1_dist(x, y):
            M_wasserstein1 = ot.dist(x, y, metric='euclidean')
            wasserstein1_dist = ot.emd2(a, b, M_wasserstein1)
            return wasserstein1_dist

        def calculate_wasserstein2_dist(x, y):
            M_wasserstein2 = ot.dist(x, y, metric='sqeuclidean')
            wasserstein2_dist = np.sqrt(ot.emd2(a, b, M_wasserstein2))
            return wasserstein2_dist

        wasserstein1_pval = permutation_test(model_samples, true_samples,
                                             func=calculate_wasserstein1_dist,
                                             method='approximate',
                                             num_rounds=n_permutations,
                                             seed=0)

        wasserstein2_pval = permutation_test(model_samples, true_samples,
                                             func=calculate_wasserstein2_dist,
                                             method='approximate',
                                             num_rounds=n_permutations,
                                             seed=0)

        results = {
            'wasserstein1 pval': wasserstein1_pval,
            'wasserstein2 pval': wasserstein2_pval,
        }

        model_samples_var = to_torch_variable(model_samples)
        true_samples_var = to_torch_variable(true_samples)

        fr = FRStatistic(n, n)
        matrix = fr(model_samples_var, true_samples_var, norm=norm, ret_matrix=True)[1]
        results['Friedman-Rafsky pval'] = fr.pval(matrix, n_permutations=n_permutations)

        knn = KNNStatistic(n, n, k)
        matrix = knn(model_samples_var, true_samples_var, norm=norm, ret_matrix=True)[1]
        results['kNN pval'] = knn.pval(matrix, n_permutations=n_permutations)

        if alphas is not None:
            mmd = MMDStatistic(n, n)
            matrix = mmd(model_samples_var, true_samples_var, alphas=None, ret_matrix=True)[1]
            results['MMD pval'] = mmd.pval(matrix, n_permutations=n_permutations)

        energy = EnergyStatistic(n, n)
        matrix = energy(model_samples_var, true_samples_var, ret_matrix=True)[1]
        results['Energy pval'] = energy.pval(matrix, n_permutations=n_permutations)

        return results
