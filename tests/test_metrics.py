import pytest
from pytest import approx

from experiments.evaluation import calculate_metrics
from models.linear import LinearGenModel
from data.synthetic import generate_wty_linear_multi_w_data
from causal_estimators.ipw_estimator import IPWEstimator
from causal_estimators.standardization_estimator import StandardizationEstimator
from utils import class_name

ATE = 5
N = 50


@pytest.fixture('module')
def linear_gen_model():
    w, t, y = generate_wty_linear_multi_w_data(n=N, wdim=5, binary_treatment=True, delta=ATE, data_format='numpy')
    return LinearGenModel(w, t, y, binary_treatment=True)


@pytest.fixture('module', params=[IPWEstimator(), StandardizationEstimator()], ids=class_name)
def estimator(request):
    return request.param


def test_ate_metrics(linear_gen_model, estimator):
    metrics = calculate_metrics(gen_model=linear_gen_model, estimator=estimator, n_seeds=10, conf_ints=False)
    assert metrics['ate_squared_bias'] + metrics['ate_variance'] == approx(metrics['ate_mse'])
    assert metrics['ate_bias'] == approx(0, abs=1)


def test_mean_ite_metrics(linear_gen_model):
    metrics = calculate_metrics(gen_model=linear_gen_model, estimator=StandardizationEstimator(),
                                n_seeds=10, conf_ints=False)
    assert metrics['mean_ite_abs_bias'] == approx(0, abs=1.3)
    assert metrics['mean_ite_mse'] == approx(metrics['mean_pehe_squared'])


def test_vector_ite_metrics(linear_gen_model):
    metrics = calculate_metrics(gen_model=linear_gen_model, estimator=StandardizationEstimator(),
                                n_seeds=10, conf_ints=False, return_ite_vectors=True)
    assert metrics['ite_squared_bias'] + metrics['ite_variance'] == approx(metrics['ite_mse'])
