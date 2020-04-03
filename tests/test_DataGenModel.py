import pytest
from pytest import approx

from pyro.infer.autoguide import AutoNormal
from DataGenModel import DataGenModel
from data.synthetic import generate_zty_linear_scalar_data, generate_zty_linear_multi_z_data
from models import linear_gaussian_full_model, linear_gaussian_outcome_model, linear_multi_z_outcome_model
from utils import PANDAS, TORCH


@pytest.fixture(scope='module', params=[PANDAS, TORCH])
def linear_scalar_data(request):
    return generate_zty_linear_scalar_data(100, data_format=request.param, alpha=2, beta=10, delta=5)


@pytest.mark.parametrize('model', [
    linear_gaussian_full_model, linear_gaussian_outcome_model,
])
def test_fast_scalar_model(linear_scalar_data, model):
    DataGenModel(linear_scalar_data, model, AutoNormal, n_iters=10)


@pytest.fixture(scope='module')
def linear_multi_z_data():
    return generate_zty_linear_multi_z_data(100, zdim=10)


def test_fast_multi_z_model(linear_multi_z_data):
    DataGenModel(linear_multi_z_data, linear_multi_z_outcome_model, AutoNormal, n_iters=10)
