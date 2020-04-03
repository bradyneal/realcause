import pytest
from pytest import approx

from pyro.infer.autoguide import AutoNormal
from DataGenModel import DataGenModel
from data.synthetic import generate_zty_linear_scalar_data
from models import linear_gaussian_full_model, linear_gaussian_outcome_model


@pytest.fixture(scope='module', params=[
    (1, 0.03, 1500),
    (5, 0.03, 1500),
    (7, 0.03, 1500),
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
        return ate_est

    return _linear_gen_model_ate


@pytest.mark.slow
def test_linear_full_model_ate(linear_gen_model_ate):
    ate_est = linear_gen_model_ate(model=linear_gaussian_full_model)
    assert ate_est == approx(ate_est, abs=.1)


@pytest.mark.slow
def test_linear_outcome_model_ate(linear_gen_model_ate):
    ate_est = linear_gen_model_ate(model=linear_gaussian_outcome_model)
    assert ate_est == approx(ate_est, abs=.1)




@pytest.fixture(scope='class')
def linear_scalar_df():
    return generate_zty_linear_scalar_data(100, alpha=2, beta=10, delta=5)


@pytest.mark.parametrize('model', [
    linear_gaussian_full_model, linear_gaussian_outcome_model,
])
def test_fast_model(linear_scalar_df, model):
    DataGenModel(linear_scalar_df, model, AutoNormal, n_iters=10)