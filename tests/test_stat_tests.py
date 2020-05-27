import pytest

from models.linear import LinearGenModel
from data.synthetic import generate_wty_linear_multi_w_data
from data.lalonde import load_lalonde

ATE = 5
N = 50


@pytest.fixture(scope='module')
def linear_gen_model():
    ate = ATE
    w, t, y = generate_wty_linear_multi_w_data(N, data_format='numpy', wdim=5, delta=ate)
    return LinearGenModel(w, t, y)


@pytest.fixture(scope='module')
def lalonde_linear_gen_model():
    w, t, y = load_lalonde()
    return LinearGenModel(w, t, y)


def test_null_true_linear_data_linear_model(linear_gen_model):
    uni_metrics = linear_gen_model.get_univariate_quant_metrics()
    multi_metrics = linear_gen_model.get_multivariate_quant_metrics()
    metrics = {**uni_metrics, **multi_metrics}
    for k, v in metrics.items():
        if 'pval' in k:
            print(k, v)
            assert v > 0.2


# NOTE: this took about 1.5 hours on my 2014 Macbook Air
@pytest.mark.slow
def test_power_linear_model_lalonde(lalonde_linear_gen_model):
    metrics = lalonde_linear_gen_model.get_multivariate_quant_metrics()
    for k, v in metrics.items():
        if 'pval' in k:
            print(k, v)
            assert v < 0.01
