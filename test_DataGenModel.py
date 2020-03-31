import pytest
from pytest import approx

from pyro.infer.autoguide import AutoNormal
from DataGenModel import DataGenModel
from data.synthetic import generate_zty_linear_scalar_data
from models import linear_gaussian_full_model


# @pytest.fixture(scope='session')
# def delta7_df():
#     return generate_zty_linear_scalar_data(100, alpha=2, beta=10, delta=7)
#
# @pytest.fixture(scope='session')
# def gen_model_delta7(delta7_df):
#     return DataGenModel(delta7_df, linear_gaussian_full_model, AutoNormal, n_iters=1500)

@pytest.mark.parametrize('ate, lr, n_iters', [
    (1, 0.03, 1500),
    (5, 0.03, 1500),
    (7, 0.03, 1500),
    (20, 0.05, 3000),
    (0, 0.03, 1000),
    (-5, 0.03, 1000),
    (-20, 0.05, 2500),
])
def test_linear_ate(ate, lr, n_iters):
    df = generate_zty_linear_scalar_data(500, alpha=2, beta=10, delta=ate)
    gen_model = DataGenModel(df, linear_gaussian_full_model, AutoNormal, lr=lr, n_iters=n_iters)
    print('expected: {}\t actual: {}'.format(ate, gen_model.get_ate()))
    # assert gen_model.get_ate() == approx(ate, rel=.1, abs=.1)
    assert gen_model.get_ate() == approx(ate, abs=.1)
