import pytest
from pytest import approx

from pyro.infer.autoguide import AutoNormal
from DataGenModel import DataGenModel
from data.synthetic import generate_zty_linear_scalar_data
from models import linear_gaussian_full_model, linear_gaussian_outcome_model
from utils import PANDAS, TORCH


@pytest.fixture(scope='module', params=[PANDAS, TORCH])
def linear_scalar_data(request):
    return generate_zty_linear_scalar_data(100, data_format=request.param, alpha=2, beta=10, delta=5)


@pytest.mark.parametrize('model', [
    linear_gaussian_full_model, linear_gaussian_outcome_model,
])
def test_fast_model(linear_scalar_data, model):
    DataGenModel(linear_scalar_data, model, AutoNormal, n_iters=10)
