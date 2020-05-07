from abc import ABCMeta, abstractmethod
from numbers import Number
import numpy as np
from scipy import stats

from plotting import compare_joints, compare_bivariate_marginals
from utils import T, Y, to_np_vectors

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

    def sample(self):
        w = self.sample_w()
        t = self.sample_t(w)
        y = self.sample_y(w, t)
        return w, t, y

    def sample_interventional(self, t, w=None):
        if w is None:
            w = self.sample_w()
        if not isinstance(w, np.ndarray):
            raise ValueError('Unsupported data type: {} ... only numpy is currently supported'.format(type(w)))
        if isinstance(t, Number):
            t = np.full_like(w, t)
        return self.sample_y(t, w)

    def get_interventional_mean(self, t):
        samples = self.sample_interventional(t)
        return samples.mean()

    def get_ate(self, t1=1, t0=0, w=None):
        return self.get_interventional_mean(t1, w) - self.get_interventional_mean(t0, w)

    def get_ite(self, t1=1, t0=0, w=None):
        if w is None:
            w = self.sample_w()
        return self.sample_interventional(t1, w) - self.sample_interventional(t0, w)

    def plot_ty_dists(self, joint=True, marginal_hist=True, marginal_qq=True, name=None,
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
                           save_fname='{}_ty_joints.{}'.format(name, file_ext),
                           name=name, test=test, kwargs=joint_kwargs)

        if marginal_hist or marginal_qq:
            compare_bivariate_marginals(t_true, t_model, y_true, y_model,
                                        xlabel=T, ylabel=Y,
                                        label1=TRUE_LABEL, label2=MODEL_LABEL,
                                        hist=marginal_hist, qqplot=marginal_qq,
                                        save_hist_fname='{}_ty_marginal_hists.{}'.format(name, file_ext),
                                        save_qq_fname='{}_ty_marginal_qqplots.{}'.format(name, file_ext),
                                        name=name, test=test)

    def get_univariate_quant_metrics(self, thin_model=None, thin_true=None):
        """
        Calculates quantitative metrics for the difference between p(t) and
        p_model(t) and the difference between p(y) and p_model(y)

        :param thin_model: thinning interval for the model data
        :param thin_true: thinning interval for the real data
        :return: {
            't_ks_pval': ks p-value with null that t_model and t_true are from the same distribution
            'y_ks_pval': ks p-value with null that y_model and y_true are from the same distribution
            't_wasserstein_dist': wasserstein distance between t_true and t_model
            'y_wasserstein_dist': wasserstein distance between y_true and y_model
        }
        """
        _, t_model, y_model = to_np_vectors(self.sample(), thin_interval=thin_model)
        t_true, y_true = to_np_vectors((self.t, self.y), thin_interval=thin_true)
        ks_label = '_ks_pval'
        wasserstein_label = '_wasserstein_dist'
        metrics = {
            T + ks_label: stats.ks_2samp(t_model, t_true).pvalue,
            Y + ks_label: stats.ks_2samp(y_model, y_true).pvalue,
            T + wasserstein_label: stats.wasserstein_distance(t_model, t_true),
            Y + wasserstein_label: stats.wasserstein_distance(y_model, y_true),
        }
        return metrics
