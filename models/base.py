from abc import ABCMeta, abstractmethod
from numbers import Number
import numpy as np
import torch
from scipy import stats
import warnings
from typing import Type

from models.preprocess import Preprocess, PlaceHolderTransform
from plotting.plotting import compare_joints, compare_bivariate_marginals
from utils import T, Y, to_np_vectors, to_np_vector, to_torch_variable, permutation_test, regular_round

MODEL_LABEL = "model"
TRUE_LABEL = "true"
T_MODEL_LABEL = "{} ({})".format(T, MODEL_LABEL)
Y_MODEL_LABEL = "{} ({})".format(Y, MODEL_LABEL)
T_TRUE_LABEL = "{} ({})".format(T, TRUE_LABEL)
Y_TRUE_LABEL = "{} ({})".format(Y, TRUE_LABEL)
SEED = 42
TRAIN = "train"
VAL = "val"
TEST = "test"


class BaseGenModelMeta(ABCMeta):
    """
    Forces subclasses to implement abstract_attributes
    """

    abstract_attributes = []

    def __call__(cls, *args, **kwargs):
        obj = super(BaseGenModelMeta, cls).__call__(*args, **kwargs)
        missing_attributes = []
        for attr_name in obj.abstract_attributes:
            if not hasattr(obj, attr_name):
                missing_attributes.append(attr_name)
        if len(missing_attributes) == 1:
            raise TypeError(
                "Can't instantiate abstract class {} with abstract attribute '{}'. "
                "You must set self.{} in the constructor.".format(
                    cls.__name__, missing_attributes[0], missing_attributes[0]
                )
            )
        elif len(missing_attributes) > 1:
            raise TypeError(
                "Can't instantiate abstract class {} with abstract attributes {}. "
                "For example, you must set self.{} in the constructor.".format(
                    cls.__name__, missing_attributes, missing_attributes[0]
                )
            )

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

    abstract_attributes = ['w', 't', 'y',
                           'w_transformed', 't_transformed', 'y_transformed']

    def __init__(self, w, t, y, train_prop=1, val_prop=0, test_prop=0,
                 test_size=None, shuffle=True, seed=SEED,
                 w_transform: Type[Preprocess] = PlaceHolderTransform,
                 t_transform: Type[Preprocess] = PlaceHolderTransform,
                 y_transform: Type[Preprocess] = PlaceHolderTransform,
                 verbose=True):
        """
        Initialize the generative model. Split the data up according to the
        splits specified by train_prop, val_prop, and test_prop. These can add
        to 1, or they can just be arbitary numbers that correspond to the
        unnormalized fraction of the dataset for each subsample.
        :param w: ndarray of covariates
        :param t: ndarray for vector of treatment
        :param y: ndarray for vector of outcome
        :param train_prop: number to use for proportion of the whole dataset
            that is in the training set
        :param val_prop: number to use for proportion of the whole dataset that
            is in the validation set
        :param test_prop: number to use for proportion of the whole dataset that
            is in the test set
        :param test_size: size of the test set
        :param shuffle: boolean for whether to shuffle the data
        :param seed: random seed for pytorch and numpy
        :param w_transform: transform for covariates
        :param t_transform: transform for treatment
        :param y_transform: transform for outcome
        :param verbose: boolean
        """
        if seed is not None:
            self.set_seed(seed)

        # Split the data into train and test

        n = w.shape[0]
        idxs = np.arange(n)

        if test_size is None:
            total = train_prop + val_prop + test_prop
            n_train = regular_round(n * train_prop / total)
            n_val = regular_round(n * val_prop / total)
            n_test = n - n_train - n_val
        else:
            total = train_prop + val_prop
            n = n - test_size
            n_train = regular_round(n * train_prop / total)
            n_val = regular_round(n * val_prop / total)
            n_test = test_size

        if verbose:
            print("n_train: {}\tn_val: {}\tn_test: {}".format(n_train, n_val, n_test))

        if shuffle:
            np.random.shuffle(idxs)
        train_idxs = idxs[:n_train]
        val_idxs = idxs[n_train:n_train + n_val]
        test_idxs = idxs[n_train + n_val:]

        print("test_idxs: ", test_idxs.shape)

        self.train_idxs = train_idxs
        self.val_idxs = val_idxs
        self.test_idxs = test_idxs
        self.w = w[train_idxs]
        self.t = t[train_idxs]
        self.y = y[train_idxs]
        self.w_val = w[val_idxs]
        self.t_val = t[val_idxs]
        self.y_val = y[val_idxs]
        self.w_test = w[test_idxs]
        self.t_test = t[test_idxs]
        self.y_test = y[test_idxs]

        # Transforms computed only on the training set
        self.w_transform = w_transform(w[train_idxs])
        self.t_transform = t_transform(t[train_idxs])
        self.y_transform = y_transform(y[train_idxs])

        self.w_transformed = self.w_transform.transform(self.w)
        self.t_transformed = self.t_transform.transform(self.t)
        self.y_transformed = self.y_transform.transform(self.y)
        self.w_val_transformed = self.w_transform.transform(self.w_val)
        self.t_val_transformed = self.t_transform.transform(self.t_val)
        self.y_val_transformed = self.y_transform.transform(self.y_val)
        self.w_test_transformed = self.w_transform.transform(self.w_test)
        self.t_test_transformed = self.t_transform.transform(self.t_test)
        self.y_test_transformed = self.y_transform.transform(self.y_test)

    def get_data(self, transformed=False, dataset=TRAIN, verbose=True):
        """
        Get the specific dataset. Splits were determined in the constructor.

        :param transformed: If True, use transformed version of data.
            If False, use original (non-transformed) version of data.
        :param dataset: dataset subset to use (train, val, or test)
        :param verbose:
        :return: (covariates, treatment, outcome)
        """
        dataset = dataset.lower()
        if dataset == TRAIN:
            w, t, y = self.w, self.t, self.y
        elif dataset == VAL or dataset == "validation":
            w, t, y = self.w_val, self.t_val, self.y_val
        elif dataset == TEST:
            w, t, y = self.w_test, self.t_test, self.y_test
        else:
            raise ValueError("Invalid dataset: {}".format(dataset))

        assert w.shape[0] == t.shape[0] == y.shape[0]
        if w.shape[0] == 0:
            raise ValueError('Dataset "{}" has 0 examples in it. Please '
                             'increase the value of the corresponding argument '
                             'in the constructor.'.format(dataset))

        if transformed:
            w = self.w_transform.transform(w)
            t = self.t_transform.transform(t)
            y = self.y_transform.transform(y)
            if verbose:
                warnings.warn("transformed")
        else:
            if verbose:
                warnings.warn("untransformed")

        return w, t, y

    def sample_w(self, untransform=True, seed=None, dataset=TRAIN):
        if seed is not None:
            self.set_seed(seed)
        if untransform:
            if dataset == TEST:
                return self.w_test
            elif dataset == VAL:
                return self.w_val
            else:
                return self.w
                    
        else:
            if dataset == TEST:
                return self.w_test_transformed
            elif dataset == VAL:
                return self.w_val_transformed
            else:
                return self.w_transformed



    @abstractmethod
    def _sample_t(self, w, overlap=1):
        pass

    @abstractmethod
    def _sample_y(self, t, w, ret_counterfactuals=False):
        pass

    @abstractmethod
    def mean_y(self, t, w):
        pass

    def sample_t(self, w, untransform=True, overlap=1, seed=None):
        """
        Sample the treatment vector.

        :param w: covariate (confounder)
        :param untransform: whether to transform the data back to the raw scale
        :param overlap: if 1, leave treatment untouched;
            if 0, push p(T = 1 | w) to 0 for all w where p(T = 1 | w) < 0.5 and
            and push p(T = 1 | w) to 1 for all w where p(T = 1 | w) >= 0.5
            if 0 < overlap < 1, do a linear interpolation of the above
        :param seed: random seed
        :return: sampled treatment
        """
        if seed is not None:
            self.set_seed(seed)
        if w is None:
            # note: input to the model need to be transformed
            w = self.sample_w(untransform=False)
        t = self._sample_t(w, overlap=overlap)
        if untransform:
            return self.t_transform.untransform(t)
        else:
            return t

    def sample_y(self, t, w, untransform=True, causal_effect_scale=None,
                 deg_hetero=1.0, ret_counterfactuals=False, seed=None):
        """
        :param t: treatment
        :param w: covariate (confounder)
        :param untransform: whether to transform the data back to the raw scale
        :param causal_effect_scale: scale of the causal effect (size of ATE)
        :param deg_hetero: degree of heterogeneity (between 0 and 1)
            When deg_hetero=1, y1 and y0 remain unchanged. When deg_hetero=0,
            y1 - y0 is the same for all individuals.
        :param ret_counterfactuals: return counterfactuals if True
        :param seed: random seed
        :return: sampled outcome
        """
        if seed is not None:
            self.set_seed(seed)
        if w is None:
            # note: input to the model need to be transformed
            w = self.sample_w(untransform=False)

        y0, y1 = self._sample_y(t, w, ret_counterfactuals=True)
        if untransform:
            y0 = self.y_transform.untransform(y0)
            y1 = self.y_transform.untransform(y1)

        if deg_hetero == 1.0 and causal_effect_scale == None:  # don't change heterogeneity or causal effect size
            pass
        else:   # change degree of heterogeneity and/or causal effect size
            # degree of heterogeneity
            if deg_hetero != 1.0:
                assert 0 <= deg_hetero < 1, f'deg_hetero not in [0, 1], got {deg_hetero}'
                y1_mean = y1.mean()
                y0_mean = y0.mean()
                ate = y1_mean - y0_mean

                # calculate value to shrink either y1 or y0 (whichever is
                # further from its mean) to when deg_hetero = 0
                further_y1 = np.greater(np.abs(y1 - y1_mean), np.abs(y0 - y0_mean))
                further_y0 = np.logical_not(further_y1)
                alpha = np.random.rand(len(y1))  # how far to shrink y1 toward y1_mean or y0 toward y0_mean
                y1_limit = further_y1 * ((1 - alpha) * y1 + alpha * y1_mean)
                y0_limit = further_y0 * ((1 - alpha) * y0 + alpha * y0_mean)

                # shrink y1 (or y0) and calculate corresponding y0 or (y1) based on
                scaled_y1 = (1 - deg_hetero) * y1_limit + deg_hetero * y1 * further_y1
                corresponding_y0 = (1 - deg_hetero) * (scaled_y1 - ate) + deg_hetero * y0 * further_y1
                scaled_y0 = (1 - deg_hetero) * y0_limit + deg_hetero * y0 * further_y0
                corresponding_y1 = (1 - deg_hetero) * (scaled_y0 + ate) + deg_hetero * y1 * further_y0
                y1 = scaled_y1 * further_y1 + corresponding_y1 * further_y0
                y0 = scaled_y0 * further_y0 + corresponding_y0 * further_y1

            # size of causal effect
            if causal_effect_scale is not None:
                ate = (y1 - y0).mean()
                y1 = causal_effect_scale / ate * y1
                y0 = causal_effect_scale / ate * y0

        if ret_counterfactuals:
            return y0, y1
        else:
            return y0 * (1 - t) + y1 * t

    def set_seed(self, seed=SEED):
        torch.manual_seed(seed)
        np.random.seed(seed)

    def sample(self, w=None, transform_w=True, untransform=True, seed=None, dataset=TRAIN, overlap=1,
               causal_effect_scale=None, deg_hetero=1.0, ret_counterfactuals=False):
        """
        Sample from generative model.

        :param w: covariates (confounders)
        :param transform_w: whether to transform the w (if given)
        :param untransform: whether to transform the data back to the raw scale
        :param seed: random seed
        :param dataset: train or test for sampling w from
        :param overlap: if 1, leave treatment untouched;
            if 0, push p(T = 1 | w) to 0 for all w where p(T = 1 | w) < 0.5 and
            and push p(T = 1 | w) to 1 for all w where p(T = 1 | w) >= 0.5
            if 0 < overlap < 1, do a linear interpolation of the above
        :param causal_effect_scale: scale of the causal effect (size of ATE)
        :param deg_hetero: degree of heterogeneity (between 0 and 1)
            When deg_hetero=1, y1 and y0 remain unchanged. When deg_hetero=0,
            y1 - y0 is the same for all individuals.
        :param ret_counterfactuals: return counterfactuals if True
        :return: (w, t, y)
        """
        if seed is not None:
            self.set_seed(seed)
        if w is None:
            w = self.sample_w(untransform=False, dataset=dataset)
        elif transform_w:
            w = self.w_transform.transform(w)
        t = self.sample_t(w, untransform=False, overlap=overlap)
        if ret_counterfactuals:
            y0, y1 = self.sample_y(
                t, w, untransform=False, causal_effect_scale=causal_effect_scale,
                deg_hetero=deg_hetero, ret_counterfactuals=True
            )
            if untransform:
                return (self.w_transform.untransform(w), self.t_transform.untransform(t),
                        (self.y_transform.untransform(y0), self.y_transform.untransform(y1)))
            else:
                return w, t, (y0, y1)
        else:
            y = self.sample_y(t, w, untransform=False,
                              causal_effect_scale=causal_effect_scale,
                              deg_hetero=deg_hetero, ret_counterfactuals=False)
            if untransform:
                return self.w_transform.untransform(w), self.t_transform.untransform(t), self.y_transform.untransform(y)
            else:
                return w, t, y

    def sample_interventional(self, t, w=None, seed=None, causal_effect_scale=None, deg_hetero=1.0):
        if seed is not None:
            self.set_seed(seed)
        if w is None:
            w = self.sample_w(untransform=False)
        if isinstance(w, Number):
            raise ValueError('Unsupported data type: {} ... only numpy is currently supported'.format(type(w)))
        if isinstance(t, Number):
            t = np.full_like(self.t, t)
        return self.sample_y(t, w, causal_effect_scale=causal_effect_scale, deg_hetero=deg_hetero)

    def ate(self, t1=1, t0=0, w=None, noisy=True, untransform=True, transform_t=True, n_y_per_w=100,
            causal_effect_scale=None, deg_hetero=1.0):
        return self.ite(t1=t1, t0=t0, w=w, noisy=noisy, untransform=untransform,
                        transform_t=transform_t, n_y_per_w=n_y_per_w,
                        causal_effect_scale=causal_effect_scale,
                        deg_hetero=deg_hetero).mean()

    def noisy_ate(self, t1=1, t0=0, w=None, n_y_per_w=100, seed=None, transform_w=False):
        if w is not None and transform_w:
            w = self.w_transform.transform(w)

        # Note: bad things happen if w is not transformed and transform_w is False

        if seed is not None:
            self.set_seed(seed)

        if (isinstance(t1, Number) or isinstance(t0, Number)) and w is not None:
            t_shape = list(self.t.shape)
            t_shape[0] = w.shape[0]
            t1 = np.full(t_shape, t1)
            t0 = np.full(t_shape, t0)
        total = 0
        for _ in range(n_y_per_w):
            total += (self.sample_interventional(t=t1, w=w) -
                      self.sample_interventional(t=t0, w=w)).mean()
        return total / n_y_per_w

    def att(self, t1=1, t0=0, w=None, untransform=True, transform_t=True):
        pass
        # TODO
        # return self.ite(t1=t1, t0=t0, w=w, untransform=untransform,
        #                 transform_t=transform_t).mean()

    def ite(self, t1=1, t0=0, w=None, t=None, untransform=True, transform_t=True, transform_w=True,
            estimand="all", noisy=True, seed=None, n_y_per_w=100,
            causal_effect_scale=None, deg_hetero=1.0):
        if seed is not None:
            self.set_seed(seed)
        if w is None:
            # w = self.w_transformed
            w = self.sample_w(untransform=False)
            t = self.t
        estimand = estimand.lower()
        if estimand == "all" or estimand == "ate":
            pass
        elif estimand == "treated" or estimand == "att":
            w = w[t == 1]
        elif estimand == "control" or estimand == "atc":
            w = w[t == 0]
        else:
            raise ValueError("Invalid estimand: {}".format(estimand))
        if transform_t:
            t1 = self.t_transform.transform(t1)
            t0 = self.t_transform.transform(t0)
            # Note: check that this is an identity transformation
        if isinstance(t1, Number) or isinstance(t0, Number):
            t_shape = list(self.t.shape)
            t_shape[0] = w.shape[0]
            t1 = np.full(t_shape, t1)
            t0 = np.full(t_shape, t0)
        # if transform_w:
        #     w = self.w_transform.transform(w)
        if noisy:
            y1_total = np.zeros(w.shape[0])
            y0_total = np.zeros(w.shape[0])
            for _ in range(n_y_per_w):
                y1_total += to_np_vector(self.sample_interventional(
                    t=t1, w=w,causal_effect_scale=causal_effect_scale, deg_hetero=deg_hetero))
                y0_total += to_np_vector(self.sample_interventional(
                    t=t0, w=w, causal_effect_scale=causal_effect_scale, deg_hetero=deg_hetero))
            y_1 = y1_total / n_y_per_w
            y_0 = y0_total / n_y_per_w
        else:
            if causal_effect_scale is not None or deg_hetero != 1.0:
                raise ValueError('Invalid causal_effect_scale or deg_hetero. '
                                 'Current mean_y only supports defaults.')
            y_1 = to_np_vector(self.mean_y(t=t1, w=w))
            y_0 = to_np_vector(self.mean_y(t=t0, w=w))
        # This is already done in sample_interventional --> sample_y
        # TODO: add this argument to sample_interventional and pass it to sample_y
        # if untransform:
        #     y_1 = self.y_transform.untransform(y_1)
        #     y_0 = self.y_transform.untransform(y_0)
        return y_1 - y_0

    def plot_ty_dists(self, joint=True, marginal_hist=True, marginal_qq=True,
                      dataset=TRAIN, transformed=False, verbose=True,
                      title=True, name=None, file_ext='pdf', thin_model=None,
                      thin_true=None, joint_kwargs={}, test=False, seed=None):
        """
        Creates up to 3 different plots of the real data and the corresponding model

        :param joint: boolean for whether to plot p(t, y)
        :param marginal_hist: boolean for whether to plot the p(t) and p(y) histograms
        :param marginal_qq: boolean for whether to plot the p(y) Q-Q plot
            or use 'both' for plotting both the p(t) and p(y) Q-Q plots
        :param dataset: dataset subset to use (train, val, or test)
        :param transformed: If True, use transformed version of data.
            If False, use original (non-transformed) version of data.
        :param title: boolean for whether or not to include title in plots
        :param name: name to use in plot titles and saved files defaults to name of class
        :param file_ext: file extension to for saving plots (e.g. 'pdf', 'png', etc.)
        :param thin_model: thinning interval for the model data
        :param thin_true: thinning interval for the real data
        :param joint_kwargs: kwargs passed to sns.kdeplot() for p(t, y)
        :param test: if True, does not show or save plots
        :param seed: seed for sample from generative model
        :return:
        """
        if name is None:
            name = self.__class__.__name__

        _, t_model, y_model = to_np_vectors(self.sample(seed=seed, untransform=(not transformed)),
                                            thin_interval=thin_model)
        _, t_true, y_true = self.get_data(transformed=transformed, dataset=dataset, verbose=verbose)
        t_true, y_true = to_np_vectors((t_true, y_true), thin_interval=thin_true)
        plots = []

        if joint:
            fig1 = compare_joints(t_model, y_model, t_true, y_true,
                           xlabel1=T_MODEL_LABEL, ylabel1=Y_MODEL_LABEL,
                           xlabel2=T_TRUE_LABEL, ylabel2=Y_TRUE_LABEL,
                           xlabel=T, ylabel=Y,
                           label1=MODEL_LABEL, label2=TRUE_LABEL,
                           save_fname='{}_ty_joints.{}'.format(name, file_ext),
                           title=title, name=name, test=test, kwargs=joint_kwargs)
            
            plots += [fig1]

        if marginal_hist or marginal_qq:
            plots += compare_bivariate_marginals(t_true, t_model, y_true, y_model,
                                        xlabel=T, ylabel=Y,
                                        label1=TRUE_LABEL, label2=MODEL_LABEL,
                                        hist=marginal_hist, qqplot=marginal_qq,
                                        save_hist_fname='{}_ty_marginal_hists.{}'.format(name, file_ext),
                                        save_qq_fname='{}_ty_marginal_qqplots.{}'.format(name, file_ext),
                                        title=title, name=name, test=test)
            
        return plots

    def get_univariate_quant_metrics(self, dataset=TRAIN, transformed=False, verbose=True,
                                     thin_model=None, thin_true=None, seed=None, n=None):
        """
        Calculates quantitative metrics for the difference between p(t) and
        p_model(t) and the difference between p(y) and p_model(y)

        :param dataset: dataset subset to evaluate on (train, val, or test)
        :param transformed: If True, use transformed version of data.
            If False, use original (non-transformed) version of data.
        :param thin_model: thinning interval for the model data
        :param thin_true: thinning interval for the real data
        :param seed: seed for sample from generative model
        :return: {
            't_ks_pval': ks p-value with null that t_model and t_true are from the same distribution
            'y_ks_pval': ks p-value with null that y_model and y_true are from the same distribution
            't_wasserstein1_dist': wasserstein1 distance between t_true and t_model
            'y_wasserstein1_dist': wasserstein1 distance between y_true and y_model
        }
        """
        _, t_model, y_model = to_np_vectors(
            self.sample(seed=seed, untransform=(not transformed)),
            thin_interval=thin_model
        )

        _, t_true, y_true = self.get_data(transformed=transformed, dataset=dataset, verbose=verbose)
        t_true, y_true = to_np_vectors((t_true, y_true), thin_interval=thin_true)

        # jitter for numerical stability
        t_true = t_true.copy() + np.random.rand(*t_true.shape) * 1e-6
        t_model = t_model.copy() + np.random.rand(*t_model.shape) * 1e-6

        ks_label = "_ks_pval"
        es_label = "_es_pval"
        wasserstein_label = "_wasserstein1_dist"
        metrics = {
            T + ks_label: float(stats.ks_2samp(t_model, t_true).pvalue),
            Y + ks_label: float(stats.ks_2samp(y_model, y_true).pvalue),
            T + es_label: float(stats.epps_singleton_2samp(t_model, t_true).pvalue),
            Y + es_label: float(stats.epps_singleton_2samp(y_model, y_true).pvalue),
            T + wasserstein_label: float(stats.wasserstein_distance(t_model, t_true)),
            Y + wasserstein_label: float(stats.wasserstein_distance(y_model, y_true)),
        }

        return metrics

    def get_multivariate_quant_metrics(
        self,
        include_w=True,
        dataset=TRAIN,
        transformed=False,
        norm=2,
        k=1,
        alphas=None,
        n_permutations=1000,
        seed=None,
        verbose=False,
        n=None
    ):
        """
        Computes Wasserstein-1 and Wasserstein-2 distances. Also computes all the
        test statistics and p-values for the multivariate two sample tests from
        the torch_two_sample package. See that documentation for more info on
        the specific tests: https://torch-two-sample.readthedocs.io/en/latest/

        :param include_w: If False, test if p(t, y) = p_model(t, y).
            If True, test if p(w, t, y) = p(w, t, y).
        :param dataset: dataset subset to evaluate on (train, val, or test)
        :param transformed: If True, use transformed version of data.
            If False, use original (non-transformed) version of data.
        :param norm: norm used for Friedman-Rafsky test and kNN test
        :param k: number of nearest neighbors to use for kNN test
        :param alphas: list of kernel parameters for MMD test
        :param n_permutations: number of permutations for each test
        :param seed: seed for sample from generative model
        :param verbose: print intermediate steps
        :param n: subsample dataset to n samples

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
            raise ModuleNotFoundError(
                str(e)
                + " ... Install: conda install cython && conda install -c conda-forge pot"
            )
        try:
            from torch_two_sample.statistics_nondiff import FRStatistic, KNNStatistic
            from torch_two_sample.statistics_diff import MMDStatistic, EnergyStatistic
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(str(e) + ' ... Install: pip install git+git://github.com/josipd/torch-two-sample')
        w_model, t_model, y_model = self.sample(seed=seed, untransform=(not transformed))
        if n is not None and w_model.shape[0] > n:
            select_rows = np.random.choice(w_model.shape[0], n, replace=False)
            w_model = w_model[select_rows, :]
            t_model = t_model[select_rows, :]
            y_model = y_model[select_rows, :]

        t_model, y_model = to_np_vectors((t_model, y_model), column_vector=True)
        model_samples = np.hstack((t_model, y_model))

        w_true, t_true, y_true = self.get_data(transformed=transformed, dataset=dataset, verbose=verbose)
        if n is not None and w_true.shape[0] > n:
            select_rows = np.random.choice(w_true.shape[0], n, replace=False)
            w_true = w_true[select_rows, :]
            t_true = t_true[select_rows, :]
            y_true = y_true[select_rows, :]

        t_true, y_true = to_np_vectors((t_true, y_true), column_vector=True)
        true_samples = np.hstack((t_true, y_true))

        if include_w:
            model_samples = np.hstack((w_model, model_samples))
            true_samples = np.hstack((w_true, true_samples))

        n_model = model_samples.shape[0]
        n_true = true_samples.shape[0]

        a, b = np.ones((n_model,)) / n_model, np.ones((n_true,)) / n_true  # uniform   # uniform distribution on samples

        def calculate_wasserstein1_dist(x, y):
            M_wasserstein1 = ot.dist(x, y, metric="euclidean")
            wasserstein1_dist = ot.emd2(a, b, M_wasserstein1)
            return wasserstein1_dist

        def calculate_wasserstein2_dist(x, y):
            M_wasserstein2 = ot.dist(x, y, metric="sqeuclidean")
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
            "wasserstein1 pval": wasserstein1_pval,
            "wasserstein2 pval": wasserstein2_pval,
        }

        model_samples_var = to_torch_variable(model_samples)
        true_samples_var = to_torch_variable(true_samples)

        fr = FRStatistic(n_model, n_true)
        matrix = fr(model_samples_var, true_samples_var, norm=norm, ret_matrix=True)[1]
        results["Friedman-Rafsky pval"] = fr.pval(matrix, n_permutations=n_permutations)

        knn = KNNStatistic(n_model, n_true, k)
        matrix = knn(model_samples_var, true_samples_var, norm=norm, ret_matrix=True)[1]
        results["kNN pval"] = knn.pval(matrix, n_permutations=n_permutations)

        if alphas is not None:
            mmd = MMDStatistic(n_model, n_true)
            matrix = mmd(model_samples_var, true_samples_var, alphas=None, ret_matrix=True)[1]
            results['MMD pval'] = mmd.pval(matrix, n_permutations=n_permutations)

        energy = EnergyStatistic(n_model, n_true)
        matrix = energy(model_samples_var, true_samples_var, ret_matrix=True)[1]
        results["Energy pval"] = energy.pval(matrix, n_permutations=n_permutations)

        return results
