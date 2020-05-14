import pytest
from pytest import approx
import pandas as pd

from causallib.estimation import IPW, Standardization
from causal_estimators.ipw_estimator import IPWEstimator
from causal_estimators.standardization_estimator import StandardizationEstimator
from utils import class_name

from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import SVR, LinearSVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor, AdaBoostRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


from data.synthetic import generate_wty_linear_multi_w_data

ATE = 5
N = 50


@pytest.fixture(scope='module')
def linear_data():
    w, t, y = generate_wty_linear_multi_w_data(n=N, wdim=5, binary_treatment=True, delta=ATE, data_format='pandas')
    return w, t, y


@pytest.fixture(scope='module', ids=class_name, params=[
    LogisticRegression(penalty='l2'),
    LogisticRegression(penalty='none'),
    LogisticRegression(penalty='l2', solver='liblinear'),
    LogisticRegression(penalty='l1', solver='liblinear'),
    LogisticRegression(penalty='l1', solver='saga'),

    # Below list comes from sklearn website:
    # https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
    KNeighborsClassifier(3),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
])
def ipw_estimator(request, linear_data):
    w, t, y = linear_data
    prop_score_model = request.param
    ipw = IPWEstimator(prop_score_model=prop_score_model)
    ipw.fit(w, t, y)
    return ipw


def test_ipw_matches_causallib(linear_data):
    w, t, y = linear_data
    causallib_ipw = IPW(learner=LogisticRegression())
    causallib_ipw.fit(w, t)
    potential_outcomes = causallib_ipw .estimate_population_outcome(w, t, y, treatment_values=[0, 1])
    causallib_effect = causallib_ipw.estimate_effect(potential_outcomes[1], potential_outcomes[0])[0]

    ipw = IPWEstimator()
    ipw.fit(w, t, y)
    our_effect = ipw.estimate_ate()
    assert our_effect == causallib_effect


def test_ipw_estimate_near_ate(ipw_estimator):
    # large tolerance because estimator doesn't work that well
    ate_est = ipw_estimator.estimate_ate()
    assert ate_est == approx(ATE, rel=.35)


def test_ipw_weight_trimming(linear_data):
    w, t, y = linear_data
    ipw = IPWEstimator(trim_weights=True)
    ipw.fit(w, t, y)
    assert ipw.estimate_ate() == approx(ATE, rel=.2)


def test_ipw_trim_eps(linear_data):
    w, t, y = linear_data
    ipw = IPWEstimator(trim_eps=.05)
    ipw.fit(w, t, y)
    assert ipw.estimate_ate() == approx(ATE, rel=.2)


def test_ipw_stabilized_weights(linear_data):
    w, t, y = linear_data
    ipw = IPWEstimator(stabilized=True)
    ipw.fit(w, t, y)
    assert ipw.estimate_ate() == approx(ATE, rel=.2)


def test_ipw_weight_trimming_and_stabilized_weights(linear_data):
    w, t, y = linear_data
    ipw = IPWEstimator(trim_weights=True, stabilized=True)
    ipw.fit(w, t, y)
    assert ipw.estimate_ate() == approx(ATE, rel=.2)


def test_ipw_numpy_data():
    w, t, y = generate_wty_linear_multi_w_data(n=N, wdim=5, binary_treatment=True, delta=ATE, data_format='numpy')
    ipw = IPWEstimator()
    ipw.fit(w, t, y)
    assert ipw.estimate_ate() == approx(ATE, rel=.2)


def test_standardization_matches_causallib(linear_data):
    w, t, y = linear_data
    causallib_standardization = Standardization(LinearRegression())
    causallib_standardization.fit(w, t, y)
    individual_potential_outcomes = causallib_standardization.estimate_individual_outcome(w, t)
    causallib_ite_estimates = individual_potential_outcomes[1] - individual_potential_outcomes[0]
    mean_potential_outcomes = causallib_standardization.estimate_population_outcome(w, t)
    causallib_ate_estimate = mean_potential_outcomes[1] - mean_potential_outcomes[0]

    standardization = StandardizationEstimator()
    standardization.fit(w, t, y)
    assert causallib_ate_estimate == standardization.estimate_ate()
    pd.testing.assert_series_equal(causallib_ite_estimates, standardization.estimate_ite())


@pytest.fixture(scope='module')
def standardization_estimator(linear_data):

    def parametric_estimator(outcome_model):
        w, t, y = linear_data
        standardization_estimator = StandardizationEstimator(outcome_model=outcome_model)
        standardization_estimator.fit(w, t, y)
        return standardization_estimator

    return parametric_estimator


@pytest.mark.parametrize('outcome_model', [
    LinearRegression(),
    Lasso(alpha=.1),
    Ridge(alpha=.1),
    ElasticNet(alpha=.01),
    SVR(kernel='linear'),
    LinearSVR(),
    KernelRidge(alpha=.1),
    MLPRegressor(max_iter=1000),
    MLPRegressor(alpha=1, max_iter=1000),
], ids=class_name)
def test_standardization_estimate_near_ate(standardization_estimator, outcome_model):
    ate_est = standardization_estimator(outcome_model).estimate_ate()
    assert ate_est == approx(ATE, rel=.1)


@pytest.mark.parametrize('outcome_model', [
    # KNeighborsRegressor(n_neighbors=6),
    GaussianProcessRegressor(1.0 * RBF(1.0)),
    DecisionTreeRegressor(max_depth=2),
    RandomForestRegressor(max_depth=5, n_estimators=10),
    AdaBoostRegressor(),
], ids=class_name)
def test_standardization_estimate_nearish_ate(standardization_estimator, outcome_model):
    ate_est = standardization_estimator(outcome_model).estimate_ate()
    assert ate_est == approx(ATE, rel=.25)