import pytest
from pytest import approx

from evaluation import calculate_metrics
from models.linear import LinearGenModel
from data.synthetic import generate_wty_linear_multi_w_data
from causal_estimators.ipw_estimator import IPWEstimator

ATE = 5
N = 50


def test_metrics():
    w, t, y = generate_wty_linear_multi_w_data(n=N, wdim=5, binary_treatment=True, delta=ATE, data_format='numpy')
    linear_gen_model = LinearGenModel(w, t, y, binary_treatment=True)
    metrics = calculate_metrics(gen_model=linear_gen_model, estimator=IPWEstimator(), n_iters=10, conf_ints=False)
    assert metrics['ate_squared_bias'] + metrics['ate_variance'] == approx(metrics['ate_mse'])
    assert metrics['ate_bias'] == approx(0, abs=1)
