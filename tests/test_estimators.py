import pytest
from pytest import approx
from causallib.estimation import IPW
from causal_estimators.ipw_estimator import IPWEstimator

from sklearn.linear_model import LogisticRegression

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


from data.synthetic import generate_wty_linear_multi_w_data

ATE = 5
N = 50


@pytest.fixture(scope='module')
def linear_data():
    ate = ATE
    w, t, y = generate_wty_linear_multi_w_data(n=100, wdim=5, binary_treatment=True, delta=5, data_format='pandas')
    return w, t, y


@pytest.fixture(scope='module', params=[
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


@pytest.fixture(scope='module')
def ipw_estimator_logistic_regression(linear_data):
    w, t, y = linear_data
    ipw = IPWEstimator()
    ipw.fit(w, t, y)
    return ipw


def test_ipw_matches_causallib(linear_data, ipw_estimator_logistic_regression):
    w, t, y = linear_data
    w, t, y = generate_wty_linear_multi_w_data(n=100, wdim=5, binary_treatment=True, delta=5, data_format='pandas')
    causallib_ipw = IPW(learner=LogisticRegression())
    causallib_ipw.fit(w, t)
    potential_outcomes = causallib_ipw .estimate_population_outcome(w, t, y, treatment_values=[0, 1])
    causallib_effect = causallib_ipw.estimate_effect(potential_outcomes[1], potential_outcomes[0])[0]

    our_effect = ipw_estimator_logistic_regression.estimate_ate()
    assert our_effect == causallib_effect


def test_ipw_estimate_near_ate(ipw_estimator):
    # large tolerance because estimator doesn't work that well
    ate_est = ipw_estimator.estimate_ate()
    assert ate_est == approx(ATE, rel=.35)
