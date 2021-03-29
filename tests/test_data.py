import pytest
from pytest import approx
import numpy as np
import pandas as pd
import torch

from data.synthetic import generate_wty_linear_scalar_data, generate_wty_linear_multi_w_data
from data.lalonde import load_lalonde
from data.ihdp import load_ihdp_datasets
from utils import NUMPY, PANDAS, PANDAS_SINGLE, TORCH


def test_linear_scalar_data_pandas():
    w, t, y = generate_wty_linear_scalar_data(10, data_format=PANDAS)
    assert isinstance(w, pd.DataFrame) and isinstance(t, pd.Series) and isinstance(y, pd.Series)


def test_linear_scalar_data_torch():
    w, t, y = generate_wty_linear_scalar_data(10, data_format=TORCH)
    assert all(isinstance(x, torch.Tensor) for x in (w, t, y))


def test_lalonde_pandas():
    w, t, y = load_lalonde(data_format=PANDAS)
    assert isinstance(w, pd.DataFrame) and isinstance(t, pd.Series) and isinstance(y, pd.Series)


def test_lalonde_pandas_single():
    df = load_lalonde(data_format=PANDAS_SINGLE)
    assert isinstance(df, pd.DataFrame)


def test_lalonde_torch():
    (w, t, y) = load_lalonde(data_format=TORCH)
    assert all(isinstance(x, torch.Tensor) for x in (w, t, y))


def test_lalonde_defaults():
    w1, t1, y1 = load_lalonde()
    w2, t2, y2 = load_lalonde(rct_version='dw', obs_version='psid1', data_format=NUMPY)
    np.testing.assert_array_equal(w1, w2)
    np.testing.assert_array_equal(t1, t2)
    np.testing.assert_array_equal(y1, y2)


def test_lalonde_dw_rct():
    (w, t, y) = load_lalonde(rct_version='dw', rct=True)
    n = 260 + 185
    assert w.shape == (n, 8)
    assert t.shape == (n,)
    assert y.shape == (n,)
    ate = y[t == 1].mean() - y[t == 0].mean()
    assert round(ate) == 1794


def test_lalonde_original_rct():
    (w, t, y) = load_lalonde(rct_version='lalonde', rct=True)
    n = 425 + 297
    assert w.shape == (n, 7)
    assert t.shape == (n,)
    assert y.shape == (n,)
    ate = y[t == 1].mean() - y[t == 0].mean()
    assert round(ate) == 886


def test_lalonde_psid1():
    (w, t, y) = load_lalonde(rct_version='dw', obs_version='psid1')
    n = 2490 + 185
    assert w.shape == (n, 8)
    assert t.shape == (n,)
    assert y.shape == (n,)
    ate = y[t == 1].mean() - y[t == 0].mean()
    assert round(ate) == -15205


def test_lalonde_cps1():
    (w, t, y) = load_lalonde(rct_version='dw', obs_version='cps1')
    n = 15992 + 185
    assert w.shape == (n, 8)
    assert t.shape == (n,)
    assert y.shape == (n,)
    ate = y[t == 1].mean() - y[t == 0].mean()
    assert round(ate) == -8498


def test_lalonde_original_psid1():
    (w, t, y) = load_lalonde(rct_version='lalonde', obs_version='psid1')
    n = 2490 + 297
    assert w.shape == (n, 7)
    assert t.shape == (n,)
    assert y.shape == (n,)
    ate = y[t == 1].mean() - y[t == 0].mean()
    assert round(ate) == -15578


def test_lalonde_original_cps1():
    (w, t, y) = load_lalonde(rct_version='lalonde', obs_version='cps1')
    n = 15992 + 297
    assert w.shape == (n, 7)
    assert t.shape == (n,)
    assert y.shape == (n,)
    ate = y[t == 1].mean() - y[t == 0].mean()
    assert round(ate) == -8870


def test_lalonde_psid2():
    (w, t, y) = load_lalonde(rct_version='dw', obs_version='psid2')
    n = 253 + 185
    assert w.shape == (n, 8)
    assert t.shape == (n,)
    assert y.shape == (n,)
    ate = y[t == 1].mean() - y[t == 0].mean()
    assert round(ate) == -3647


def test_lalonde_cps2():
    (w, t, y) = load_lalonde(rct_version='dw', obs_version='cps2')
    n = 2369 + 185
    assert w.shape == (n, 8)
    assert t.shape == (n,)
    assert y.shape == (n,)
    ate = y[t == 1].mean() - y[t == 0].mean()
    assert round(ate) == -3822


def test_lalonde_original_psid2():
    (w, t, y) = load_lalonde(rct_version='lalonde', obs_version='psid2')
    n = 253 + 297
    assert w.shape == (n, 7)
    assert t.shape == (n,)
    assert y.shape == (n,)
    ate = y[t == 1].mean() - y[t == 0].mean()
    assert round(ate) == -4020


def test_lalonde_original_cps2():
    (w, t, y) = load_lalonde(rct_version='lalonde', obs_version='cps2')
    n = 2369 + 297
    assert w.shape == (n, 7)
    assert t.shape == (n,)
    assert y.shape == (n,)
    ate = y[t == 1].mean() - y[t == 0].mean()
    assert round(ate) == -4195


def test_lalonde_psid3():
    (w, t, y) = load_lalonde(rct_version='dw', obs_version='psid3')
    n = 128 + 185
    assert w.shape == (n, 8)
    assert t.shape == (n,)
    assert y.shape == (n,)
    ate = y[t == 1].mean() - y[t == 0].mean()
    assert round(ate) == 1070


def test_lalonde_cps3():
    (w, t, y) = load_lalonde(rct_version='dw', obs_version='cps3')
    n = 429 + 185
    assert w.shape == (n, 8)
    assert t.shape == (n,)
    assert y.shape == (n,)
    ate = y[t == 1].mean() - y[t == 0].mean()
    assert round(ate) == -635


def test_lalonde_original_psid3():
    (w, t, y) = load_lalonde(rct_version='lalonde', obs_version='psid3')
    n = 128 + 297
    assert w.shape == (n, 7)
    assert t.shape == (n,)
    assert y.shape == (n,)
    ate = y[t == 1].mean() - y[t == 0].mean()
    assert round(ate) == 697


def test_lalonde_original_cps3():
    (w, t, y) = load_lalonde(rct_version='lalonde', obs_version='cps3')
    n = 429 + 297
    assert w.shape == (n, 7)
    assert t.shape == (n,)
    assert y.shape == (n,)
    ate = y[t == 1].mean() - y[t == 0].mean()
    assert round(ate) == -1008


def test_multivariate_w_data():
    n = 10
    d = 5
    w, t, y = generate_wty_linear_multi_w_data(n, wdim=5)
    assert all(isinstance(x, np.ndarray) for x in (w, t, y))
    assert w.shape == (n, d) and t.shape == (n,) and y.shape == (n,)


@pytest.mark.parametrize('split', ['train', 'test', 'all'])
@pytest.mark.parametrize('n_realizations', [100, 1000])
def test_ihdp_loading(split, n_realizations):
    load_ihdp_datasets(split=split, n_realizations=n_realizations)
