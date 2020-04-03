import pandas as pd
import torch

from data.synthetic import generate_zty_linear_scalar_data, generate_zty_linear_multi_z_data
from data.whynot_simulators import generate_lalonde_random_outcome
from utils import PANDAS, TORCH


def test_linear_scalar_data_pandas():
    df = generate_zty_linear_scalar_data(10, data_format=PANDAS)
    assert isinstance(df, pd.DataFrame)


def test_linear_scalar_data_torch():
    z, t, y = generate_zty_linear_scalar_data(10, data_format=TORCH)
    assert isinstance(z, torch.Tensor) and isinstance(t, torch.Tensor) and isinstance(y, torch.Tensor)


def test_lalonde_random_outcome_data_pandas():
    df, causal_effects = generate_lalonde_random_outcome(data_format=PANDAS)
    assert isinstance(df, pd.DataFrame)


def test_lalonde_random_outcome_data_torch():
    (z, t, y), causal_effects = generate_lalonde_random_outcome(data_format=TORCH)
    assert isinstance(z, torch.Tensor) and isinstance(t, torch.Tensor) and isinstance(y, torch.Tensor)


def test_multivariate_z_data():
    z, t, y = generate_zty_linear_multi_z_data(10)
    assert all(isinstance(x, torch.Tensor) for x in (z, t, y))
