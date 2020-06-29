import pytest
from pytest import approx

from models.base import BaseGenModel
from models.linear import LinearGenModel
from data.synthetic import generate_wty_linear_multi_w_data

ATE = 5
N = 50


def test_subclass_working():

    class GenModelPassing(BaseGenModel):

        def __init__(self, w, t, y):
            self.w = w
            self.t = t
            self.y = y

        def sample_t(self, w):
            pass

        def sample_y(self, t, w):
            pass

    GenModelPassing(0, 0, 0)


def test_subclass_missing_attr():

    class GenModelMissingAttr(BaseGenModel):

        def __init__(self, w, t, y):
            self.w = w
            self.t = t

        def sample_t(self, w):
            pass

        def sample_y(self, t, w):
            pass

    with pytest.raises(TypeError):
        GenModelMissingAttr(0, 0, 0)


def test_subclass_missing_method():

    class GenModelMissingMethod(BaseGenModel):

        def __init__(self, w, t, y):
            self.w = w
            self.t = t
            self.y = y

        def sample_t(self, w):
            pass

    with pytest.raises(TypeError):
        GenModelMissingMethod(0, 0, 0)


@pytest.fixture(scope='module')
def linear_gen_model():
    w, t, y = generate_wty_linear_multi_w_data(N, data_format='numpy', wdim=5, delta=ATE)
    return LinearGenModel(w, t, y)


def test_linear_gen_model(linear_gen_model):
    pass    # just a test for the linear_gen_model fixture


def test_ate(linear_gen_model):
    ate_est = linear_gen_model.ate()
    assert ate_est == approx(ATE, abs=.2)


def test_ite(linear_gen_model):
    ite_est = linear_gen_model.ite()
    assert ite_est.shape[0] == N


@pytest.mark.plot
def test_plot_ty(linear_gen_model):
    linear_gen_model.plot_ty_dists(test=True)


def test_univariate_quant_metrics(linear_gen_model):
    linear_gen_model.get_univariate_quant_metrics(transformed=False)


def test_univariate_quant_metrics_transformed(linear_gen_model):
    linear_gen_model.get_univariate_quant_metrics(transformed=True)


def test_multivariate_quant_metrics(linear_gen_model):
    linear_gen_model.get_multivariate_quant_metrics(include_w=True, transformed=False)
    linear_gen_model.get_multivariate_quant_metrics(include_w=False, transformed=False)


def test_multivariate_quant_metrics_transformed(linear_gen_model):
    linear_gen_model.get_multivariate_quant_metrics(include_w=True, transformed=True)
    linear_gen_model.get_multivariate_quant_metrics(include_w=False, transformed=True)


@pytest.fixture(scope='module')
def linear_gen_model_train_test():
    w, t, y = generate_wty_linear_multi_w_data(N, data_format='numpy', wdim=5, delta=ATE)
    return LinearGenModel(w, t, y, train_prop=.5, test_prop=.5)


def test_univariate_quant_metrics_test_set(linear_gen_model_train_test):
    linear_gen_model_train_test.get_univariate_quant_metrics()


def test_multivariate_quant_metrics(linear_gen_model_train_test):
    linear_gen_model_train_test.get_multivariate_quant_metrics(include_w=True, transformed=False)
    linear_gen_model_train_test.get_multivariate_quant_metrics(include_w=False, transformed=False)


def test_univariate_quant_metrics_random_without_seed(linear_gen_model):
    assert linear_gen_model.get_univariate_quant_metrics() != \
           linear_gen_model.get_univariate_quant_metrics()


def test_univariate_quant_metrics_deterministic_with_seed(linear_gen_model):
    assert linear_gen_model.get_univariate_quant_metrics(seed=0) == \
           linear_gen_model.get_univariate_quant_metrics(seed=0)


def test_multivariate_quant_metrics_random_without_seed(linear_gen_model):
    assert linear_gen_model.get_multivariate_quant_metrics() != \
           linear_gen_model.get_multivariate_quant_metrics()


def test_multivariate_quant_metrics_deterministic_with_seed(linear_gen_model):
    assert linear_gen_model.get_multivariate_quant_metrics(seed=0) == \
           linear_gen_model.get_multivariate_quant_metrics(seed=0)


@pytest.fixture(scope='module')
def binary_treatment_linear_gen_model():
    w, t, y = generate_wty_linear_multi_w_data(N, binary_treatment=True, data_format='numpy', wdim=5, delta=ATE)
    return LinearGenModel(w, t, y, binary_treatment=True)


@pytest.mark.plot
def test_plot_ty_binary_treatment(binary_treatment_linear_gen_model):
    binary_treatment_linear_gen_model.plot_ty_dists(test=True)
