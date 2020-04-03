"""
Slow tests that require training whole models
"""
import pytest
from pytest import approx

from pyro.infer.autoguide import AutoNormal
from DataGenModel import DataGenModel
from data.synthetic import generate_zty_linear_scalar_data, generate_zty_linear_multi_z_data
from models import linear_gaussian_full_model, linear_gaussian_outcome_model, linear_multi_z_outcome_model

# Mark all tests in this module as slow
pytestmark = pytest.mark.slow


@pytest.fixture(scope='module', params=[
    (1, 0.03, 1500),
    (5, 0.03, 1500),
    (20, 0.05, 3000),
    (0, 0.03, 1000),
    (-5, 0.03, 1000),
    (-20, 0.05, 2500),
])
def linear_gen_model_ate(request):
    ate, lr, n_iters = request.param

    def _linear_gen_model_ate(model):
        df = generate_zty_linear_scalar_data(500, alpha=2, beta=10, delta=ate)
        gen_model = DataGenModel(df, model, AutoNormal, lr=lr, n_iters=n_iters)
        ate_est = gen_model.get_ate(n_samples_per_z=100)
        print('expected: {}\t actual: {}'.format(ate, ate_est))
        return ate, ate_est

    return _linear_gen_model_ate


def test_linear_full_model_ate(linear_gen_model_ate):
    ate, ate_est = linear_gen_model_ate(model=linear_gaussian_full_model)
    assert ate_est == approx(ate, abs=.1)


def test_linear_outcome_model_ate(linear_gen_model_ate):
    ate, ate_est = linear_gen_model_ate(model=linear_gaussian_outcome_model)
    assert ate_est == approx(ate, abs=.1)


@pytest.mark.parametrize('ate', [-5, 0, 5])
def test_linear_multi_z_outcome_model_ate(ate):
    z, t, y = generate_zty_linear_multi_z_data(500, zdim=10, delta=ate)
    gen_model = DataGenModel((z, t, y), linear_multi_z_outcome_model, AutoNormal, n_iters=1000)
    ate_est = gen_model.get_ate()
    print('expected: {}\t actual: {}'.format(ate, ate_est))
    assert ate_est == approx(ate, abs=.1)
