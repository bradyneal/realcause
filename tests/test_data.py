import pandas as pd
import torch

from data.synthetic import generate_zty_linear_scalar_data, generate_zty_linear_multi_z_data
from data.whynot_simulators import generate_lalonde_random_outcome
from data.lalonde import load_lalonde
from utils import PANDAS, TORCH

import os


def test_linear_scalar_data_pandas():
    df = generate_zty_linear_scalar_data(10, data_format=PANDAS)
    assert isinstance(df, pd.DataFrame)


def test_linear_scalar_data_torch():
    z, t, y = generate_zty_linear_scalar_data(10, data_format=TORCH)
    assert all(isinstance(x, torch.Tensor) for x in (z, t, y))


def test_lalonde_pandas():
    print(os.getcwd())
    df = load_lalonde(data_format=PANDAS)
    assert isinstance(df, pd.DataFrame)


def test_lalonde_torch():
    (w, t, y) = load_lalonde(data_format=TORCH)
    assert all(isinstance(x, torch.Tensor) for x in (w, t, y))


def test_lalonde_defaults():
    df1 = load_lalonde(data_format=PANDAS)
    df2 = load_lalonde(rct_version='dw', obs_version='psid1', data_format=PANDAS)
    pd.testing.assert_frame_equal(df1, df2)


def test_lalonde_dw_rct():
    (w, t, y) = load_lalonde(rct_version='dw', rct=True, data_format=TORCH)
    n = 260 + 185
    assert w.shape == (n, 8)
    assert t.shape == (n,)
    assert y.shape ==(n,)


def test_lalonde_original_rct():
    (w, t, y) = load_lalonde(rct_version='lalonde', rct=True, data_format=TORCH)
    n = 425 + 297
    assert w.shape == (n, 7)
    assert t.shape == (n,)
    assert y.shape ==(n,)


def test_lalonde_psid1():
    (w, t, y) = load_lalonde(rct_version='dw', obs_version='psid1', data_format=TORCH)
    n = 2490 + 185
    assert w.shape == (n, 8)
    assert t.shape == (n,)
    assert y.shape ==(n,)


def test_lalonde_cps1():
    (w, t, y) = load_lalonde(rct_version='dw', obs_version='cps1', data_format=TORCH)
    n = 15992 + 185
    assert w.shape == (n, 8)
    assert t.shape == (n,)
    assert y.shape == (n,)


def test_lalonde_original_psid1():
    (w, t, y) = load_lalonde(rct_version='lalonde', obs_version='psid1', data_format=TORCH)
    n = 2490 + 297
    assert w.shape == (n, 7)
    assert t.shape == (n,)
    assert y.shape == (n,)


def test_lalonde_original_cps1():
    (w, t, y) = load_lalonde(rct_version='lalonde', obs_version='cps1', data_format=TORCH)
    n = 15992 + 297
    assert w.shape == (n, 7)
    assert t.shape == (n,)
    assert y.shape == (n,)


def test_lalonde_random_outcome_data_pandas():
    df, causal_effects = generate_lalonde_random_outcome(data_format=PANDAS)
    assert isinstance(df, pd.DataFrame)


def test_lalonde_random_outcome_data_torch():
    (z, t, y), causal_effects = generate_lalonde_random_outcome(data_format=TORCH)
    assert all(isinstance(x, torch.Tensor) for x in (z, t, y))


def test_multivariate_z_data():
    n = 10
    d = 5
    z, t, y = generate_zty_linear_multi_z_data(n, zdim=5)
    assert all(isinstance(x, torch.Tensor) for x in (z, t, y))
    assert z.shape == (n, d) and t.shape == (n,) and y.shape == (n,)
