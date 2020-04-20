import pytest
from pytest import approx

from pyro.infer.autoguide import AutoNormal
from DataGenModel import DataGenModel
from data.synthetic import generate_zty_linear_scalar_data, generate_zty_linear_multi_z_data
from data.whynot_simulators import generate_lalonde_random_outcome
from models import linear_gaussian_full_model, linear_gaussian_outcome_model, linear_multi_z_outcome_model
from utils import PANDAS, TORCH


@pytest.fixture(scope='module', params=[PANDAS, TORCH])
def linear_scalar_data(request):
    return generate_zty_linear_scalar_data(100, data_format=request.param, alpha=2, beta=10, delta=5)


@pytest.fixture(scope='module', params=[linear_gaussian_full_model, linear_gaussian_outcome_model])
def linear_scalar_model(request):
    return request.param


@pytest.fixture(scope='module')
def linear_gen_model(linear_scalar_data, linear_scalar_model):
    return DataGenModel(linear_scalar_data, linear_scalar_model, AutoNormal, n_iters=10)


def test_fast_scalar_model(linear_gen_model):
    pass    # Just make a test for the linear_gen_model fixture


def test_fast_ate(linear_gen_model):
    linear_gen_model.get_ate()


def test_fast_plot_ty_dists(linear_gen_model):
    linear_gen_model.plot_ty_dists(n_samples_per_z=10, name='test', test=True)


@pytest.fixture(scope='module')
def linear_multi_z_data():
    return generate_zty_linear_multi_z_data(100, zdim=10)


def test_fast_multi_z_model(linear_multi_z_data):
    DataGenModel(linear_multi_z_data, linear_multi_z_outcome_model, AutoNormal, n_iters=10)


def test_fast_lalonde_linear_outcome_model(linear_multi_z_data):
    (z, t, y), causal_effects = generate_lalonde_random_outcome(data_format=TORCH)
    DataGenModel(linear_multi_z_data, linear_multi_z_outcome_model, AutoNormal, n_iters=10)
