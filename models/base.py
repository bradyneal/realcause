from abc import ABCMeta, abstractmethod
from numbers import Number
import numpy as np
import torch
from scipy import stats
from typing import Type

from models.preprocess import Preprocess, PlaceHolderTransform
from plotting import compare_joints, compare_bivariate_marginals
from utils import T, Y, to_np_vectors, to_np_vector, to_torch_variable,\
    permutation_test, regular_round

MODEL_LABEL = 'model'
TRUE_LABEL = 'true'
T_MODEL_LABEL = '{} ({})'.format(T, MODEL_LABEL)
Y_MODEL_LABEL = '{} ({})'.format(Y, MODEL_LABEL)
T_TRUE_LABEL = '{} ({})'.format(T, TRUE_LABEL)
Y_TRUE_LABEL = '{} ({})'.format(Y, TRUE_LABEL)
SEED = 42
TRAIN = 'train'
VAL = 'val'
TEST = 'test'


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

    abstract_attributes = ['w', 't', 'y',
                           'w_transformed', 't_transformed', 'y_transformed']

    def __init__(self, w, t, y, train_prop=1, val_prop=0, test_prop=0,
                 shuffle=True, seed=SEED,
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

        total = train_prop + val_prop + test_prop
        n_train = regular_round(n * train_prop / total)
        n_val = regular_round(n * val_prop / total)
        n_test = n - n_train - n_val

        if verbose:
            print('n_train: {}\tn_val: {}\tn_test: {}'.format(n_train, n_val, n_test))

        if shuffle:
            np.random.shuffle(idxs)
        train_idxs = idxs[:n_train]
        val_idxs = idxs[n_train:n_train+n_val]
        test_idxs = idxs[n_train+n_val:]

        # if train_prop is not None and train_prop < 1.0:
        #     n_train = round(train_prop * n)
        #     train_idxs = idxs[:n_train]
        #     test_idxs = idxs[n_train:]
        # else:
        #     train_idxs = idxs
        #     test_idxs = idxs

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

    def get_data(self, transformed=False, dataset=TRAIN, verbose=False):
        """
        Get the specific dataset. Splits were determined in the constructor.

        :param transformed: If True, use transformed version of data.
            If False, use original (non-transformed) version of data.
        :param dataset: dataset subset to use (train, val, or test)
        :param verbose:
        :return: (covariates, treatment, outcome)
        """
        verbose = True
        dataset = dataset.lower()
        if verbose:
            print(dataset, end=' ')
        if dataset == TRAIN:
            w, t, y = self.w, self.t, self.y
        elif dataset == VAL or dataset == 'validation':
            w, t, y = self.w_val, self.t_val, self.y_val
        elif dataset == TEST:
            w, t, y = self.w_test, self.t_test, self.y_test
        else:
            raise ValueError('Invalid dataset: {}'.format(dataset))

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
                print('transformed')
        else:
            if verbose:
                print('untransformed')

        return w, t, y

    def sample_w(self, untransform=True):
        if untransform:
            return self.w
        else:
            return self.w_transformed

    @abstractmethod
    def _sample_t(self, w):
        pass

    @abstractmethod
    def _sample_y(self, t, w):
        pass

    @abstractmethod
    def mean_y(self, t, w):
        pass

    def sample_t(self, w, untransform=True):
        if w is None:
            # note: input to the model need to be transformed
            w = self.sample_w(untransform=False)
        t = self._sample_t(w)
        if untransform:
            return self.t_transform.untransform(t)
        else:
            return t

    def sample_y(self, t, w, untransform=True):
        if w is None:
            # note: input to the model need to be transformed
            w = self.sample_w(untransform=False)
        y = self._sample_y(t, w)
        if untransform:
            return self.y_transform.untransform(y)
        else:
            return y

    def set_seed(self, seed=SEED):
        torch.manual_seed(seed)
        np.random.seed(seed)

    def sample(self, seed=None, untransform=True):
        if seed is not None:
            self.set_seed(seed)
        w = self.sample_w(untransform=False)
        t = self.sample_t(w, untransform=False)
        y = self.sample_y(t, w, untransform=False)
        if untransform:
            return self.w_transform.untransform(w), self.t_transform.untransform(t), self.y_transform.untransform(y)
        else:
            return w, t, y

    def sample_interventional(self, t, w=None):
        if w is None:
            w = self.sample_w(untransform=False)
        if isinstance(w, Number):
            raise ValueError('Unsupported data type: {} ... only numpy is currently supported'.format(type(w)))
        if isinstance(t, Number):
            t = np.full_like(self.t, t)
        return self.sample_y(t, w)

    def ate(self, t1=1, t0=0, w=None, untransform=True, transform_t=True):
        return self.ite(t1=t1, t0=t0, w=w, untransform=untransform,
                        transform_t=transform_t).mean()

    def att(self, t1=1, t0=0, w=None, untransform=True, transform_t=True):
        pass
        # TODO
        # return self.ite(t1=t1, t0=t0, w=w, untransform=untransform,
        #                 transform_t=transform_t).mean()

    def ite(self, t1=1, t0=0, w=None, t=None, untransform=True, transform_t=True, estimand='all'):
        if w is None:
            w = self.w_transformed
            # w = self.sample_w(untransform=False)
            t = self.t
        estimand = estimand.lower()
        if estimand == 'all' or estimand == 'ate':
            pass
        elif estimand == 'treated' or estimand == 'att':
            w = w[t == 1]
        elif estimand == 'control' or estimand == 'atc':
            w = w[t == 0]
        else:
            raise ValueError('Invalid estimand: {}'.format(estimand))
        if transform_t:
            t1 = self.t_transform.transform(t1)
            t0 = self.t_transform.transform(t0)
        if isinstance(t1, Number) or isinstance(t0, Number):
            t_shape = list(self.t.shape)
            t_shape[0] = w.shape[0]
            t1 = np.full(t_shape, t1)
            t0 = np.full(t_shape, t0)
        y_1 = to_np_vector(self.mean_y(t=t1, w=w))
        y_0 = to_np_vector(self.mean_y(t=t0, w=w))
        if untransform:
            y_1 = self.y_transform.untransform(y_1)
            y_0 = self.y_transform.untransform(y_0)
        return y_1 - y_0

    def plot_ty_dists(self, joint=True, marginal_hist=True, marginal_qq=True,
                      dataset=TRAIN, transformed=False,
                      title=True, name=None, file_ext='pdf', thin_model=None,
                      thin_true=None, joint_kwargs={}, test=False, seed=None):
        """
        Creates up to 3 different plots of the real data and the corresponding model

        :param joint: boolean for whether to plot p(t, y)
        :param marginal_hist: boolean for whether to plot the p(t) and p(y) histograms
        :param marginal_qq: boolean for whether to plot the p(t) and p(y) Q-Q plots
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
        _, t_true, y_true = self.get_data(transformed=transformed, dataset=dataset)
        t_true, y_true = to_np_vectors((t_true, y_true), thin_interval=thin_true)

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

    def get_univariate_quant_metrics(self, dataset=TRAIN, transformed=False,
                                     thin_model=None, thin_true=None, seed=None):
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
        _, t_model, y_model = to_np_vectors(self.sample(seed=seed, untransform=(not transformed)),
                                            thin_interval=thin_model)
        _, t_true, y_true = self.get_data(transformed=transformed, dataset=dataset)
        t_true, y_true = to_np_vectors((t_true, y_true), thin_interval=thin_true)

        ks_label = '_ks_pval'
        wasserstein_label = '_wasserstein1_dist'
        metrics = {
            T + ks_label: stats.ks_2samp(t_model, t_true).pvalue,
            Y + ks_label: stats.ks_2samp(y_model, y_true).pvalue,
            T + wasserstein_label: stats.wasserstein_distance(t_model, t_true),
            Y + wasserstein_label: stats.wasserstein_distance(y_model, y_true),
        }
        return metrics

    def get_multivariate_quant_metrics(self, include_w=True, dataset=TRAIN,
                                       transformed=False, norm=2, k=1,
                                       alphas=None, n_permutations=1000,
                                       seed=None):
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

        w_model, t_model, y_model = self.sample(seed=seed, untransform=(not transformed))
        t_model, y_model = to_np_vectors((t_model, y_model), column_vector=True)
        model_samples = np.hstack((t_model, y_model))

        w_true, t_true, y_true = self.get_data(transformed=transformed, dataset=dataset)
        t_true, y_true = to_np_vectors((t_true, y_true), column_vector=True)
        true_samples = np.hstack((t_true, y_true))

        if include_w:
            model_samples = np.hstack((w_model, model_samples))
            true_samples = np.hstack((w_true, true_samples))

        n_model = model_samples.shape[0]
        n_true = true_samples.shape[0]

        a, b = np.ones((n_model,)) / n_model, np.ones((n_true,)) / n_true  # uniform distribution on samples

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

        fr = FRStatistic(n_model, n_true)
        matrix = fr(model_samples_var, true_samples_var, norm=norm, ret_matrix=True)[1]
        results['Friedman-Rafsky pval'] = fr.pval(matrix, n_permutations=n_permutations)

        knn = KNNStatistic(n_model, n_true, k)
        matrix = knn(model_samples_var, true_samples_var, norm=norm, ret_matrix=True)[1]
        results['kNN pval'] = knn.pval(matrix, n_permutations=n_permutations)

        if alphas is not None:
            mmd = MMDStatistic(n_model, n_true)
            matrix = mmd(model_samples_var, true_samples_var, alphas=None, ret_matrix=True)[1]
            results['MMD pval'] = mmd.pval(matrix, n_permutations=n_permutations)

        energy = EnergyStatistic(n_model, n_true)
        matrix = energy(model_samples_var, true_samples_var, ret_matrix=True)[1]
        results['Energy pval'] = energy.pval(matrix, n_permutations=n_permutations)

        return results
