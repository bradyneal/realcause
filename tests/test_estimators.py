import pytest
from pytest import approx
import pandas as pd

from causallib.estimation import IPW, Standardization
from causal_estimators.ipw_estimator import IPWEstimator
from causal_estimators.standardization_estimator import \
    StandardizationEstimator, StratifiedStandardizationEstimator
from causal_estimators.doubly_robust_estimator import DoublyRobustLearner, \
    DoublyRobustEstimator, DOUBLY_ROBUST_TYPES
from causal_estimators.matching import MatchingEstimator, INVERSE_VAR, MAHALANOBIS, GENETIC
from utils import class_name
from causal_estimators.metalearners import SLearner, TLearner, XLearner
from causal_estimators.double_ml import DoubleML

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
    w, t, y = generate_wty_linear_multi_w_data(n=N, wdim=5, binary_treatment=True, delta=ATE)
    return w, t, y


@pytest.fixture(scope='module')
def linear_data_pandas():
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


def test_ipw_matches_causallib(linear_data_pandas):
    w, t, y = linear_data_pandas
    causallib_ipw = IPW(learner=LogisticRegression())
    causallib_ipw.fit(w, t)
    potential_outcomes = causallib_ipw.estimate_population_outcome(w, t, y, treatment_values=[0, 1])
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


def test_standardization_matches_causallib(linear_data_pandas):
    w, t, y = linear_data_pandas
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

    def parametric_estimator(nonparam_estimator, outcome_model):
        w, t, y = linear_data
        standardization_estimator = nonparam_estimator(outcome_model)
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
    ate_est = standardization_estimator(StandardizationEstimator, outcome_model).estimate_ate()
    assert ate_est == approx(ATE, rel=.1)


@pytest.mark.parametrize('outcome_model', [
    # KNeighborsRegressor(n_neighbors=6),
    GaussianProcessRegressor(1.0 * RBF(1.0)),
    DecisionTreeRegressor(max_depth=2),
    RandomForestRegressor(max_depth=5, n_estimators=10),
    AdaBoostRegressor(),
], ids=class_name)
def test_standardization_estimate_nearish_ate(standardization_estimator, outcome_model):
    ate_est = standardization_estimator(StandardizationEstimator, outcome_model).estimate_ate()
    assert ate_est == approx(ATE, rel=.25)


@pytest.mark.parametrize('outcome_model', [
    LinearRegression(),
    Lasso(alpha=.1),
    Ridge(alpha=.1),
    ElasticNet(alpha=.01),
    SVR(kernel='linear'),
    LinearSVR(),
    MLPRegressor(max_iter=1000),
    MLPRegressor(alpha=1, max_iter=1000),
], ids=class_name)
def test_stratified_standardization_estimate_near_ate(standardization_estimator, outcome_model):
    ate_est = standardization_estimator(StratifiedStandardizationEstimator, outcome_model).estimate_ate()
    assert ate_est == approx(ATE, rel=.1)


def test_stratified_standardization_different_models(standardization_estimator):
    outcome_models = {0: LinearRegression(), 1: ElasticNet(alpha=.01)}
    ate_est = standardization_estimator(StratifiedStandardizationEstimator, outcome_models).estimate_ate()
    assert ate_est == approx(ATE, rel=.1)


@pytest.mark.parametrize('outcome_model', [
    LinearRegression(),
    Lasso(alpha=.1),
    Ridge(alpha=.1),
    ElasticNet(alpha=.01),
    SVR(kernel='linear'),
    LinearSVR(),
    KernelRidge(alpha=.1),

    # Commented for speed of tests
    # MLPRegressor(max_iter=1000),
    # MLPRegressor(alpha=1, max_iter=1000),
], ids=class_name)
@pytest.mark.parametrize('prop_score_model', [
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
    # MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
], ids=class_name)
def test_doubly_robust_vanilla_estimate_near_ate(outcome_model, prop_score_model, linear_data):
    w, t, y = linear_data
    dr = DoublyRobustEstimator(outcome_model=outcome_model,
                               prop_score_model=prop_score_model,
                               doubly_robust_type='vanilla')
    dr.fit(w, t, y)
    assert dr.estimate_ate() == approx(ATE, rel=.1)


@pytest.mark.parametrize('outcome_model', [
    LinearRegression(),
    Lasso(alpha=.1),
    Ridge(alpha=.1),
    ElasticNet(alpha=.01),
    SVR(kernel='linear'),
    LinearSVR(),
    KernelRidge(alpha=.1),

    # Commented for speed of tests
    # MLPRegressor(max_iter=1000),
    # MLPRegressor(alpha=1, max_iter=1000),
], ids=class_name)
@pytest.mark.parametrize('prop_score_model', [
    LogisticRegression(penalty='l2'),
    LogisticRegression(penalty='none'),
    LogisticRegression(penalty='l2', solver='liblinear'),
    LogisticRegression(penalty='l1', solver='liblinear'),
    LogisticRegression(penalty='l1', solver='saga'),

    # Below list comes from sklearn website:
    # https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),

    # Commented because it gives bad ATE estimates
    # AdaBoostClassifier(),

    # Commented because the below give NaNs
    # KNeighborsClassifier(3),
    # DecisionTreeClassifier(max_depth=5),
    # RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    # MLPClassifier(alpha=1, max_iter=1000),
], ids=class_name)
def test_doubly_robust_ipfeature_estimate_near_ate(outcome_model, prop_score_model, linear_data):
    w, t, y = linear_data
    dr = DoublyRobustEstimator(outcome_model=outcome_model,
                               prop_score_model=prop_score_model,
                               doubly_robust_type='ipfeature')
    dr.fit(w, t, y)
    assert dr.estimate_ate() == approx(ATE, rel=.1)


@pytest.mark.parametrize('outcome_model', [
    LinearRegression(),
    Ridge(alpha=.1),
    SVR(kernel='linear'),
    LinearSVR(),
    KernelRidge(alpha=.1),

    # The below models' fit() methods do not support the 'sample_weight' keyword argument
    # Lasso(alpha=.1),
    # ElasticNet(alpha=.01),
    # MLPRegressor(max_iter=1000),
    # MLPRegressor(alpha=1, max_iter=1000),
], ids=class_name)
@pytest.mark.parametrize('prop_score_model', [
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
    # MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
], ids=class_name)
def test_doubly_robust_joffe_estimate_near_ate(outcome_model, prop_score_model, linear_data):
    w, t, y = linear_data
    dr = DoublyRobustEstimator(outcome_model=outcome_model,
                               prop_score_model=prop_score_model,
                               doubly_robust_type='joffe')
    dr.fit(w, t, y)
    assert dr.estimate_ate() == approx(ATE, rel=.1)


@pytest.mark.parametrize('weighting', [INVERSE_VAR, MAHALANOBIS, GENETIC])
def test_vanilla_matching(weighting, linear_data):
    w, t, y = linear_data
    match = MatchingEstimator(weighting=weighting)
    match.fit(w, t, y)
    assert match.estimate_ate() == approx(ATE, abs=1)
    lower, upper = match.ate_conf_int()
    print(match.estimate_ate(), (lower, upper))
    assert lower < ATE < upper


@pytest.mark.parametrize('prop_score_model', [
    LogisticRegression(penalty='l2'),
    LogisticRegression(penalty='none'),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
],  ids=class_name)
@pytest.mark.parametrize('weighting', [INVERSE_VAR, MAHALANOBIS, GENETIC])
def test_prop_score_matching(prop_score_model, weighting, linear_data):
    w, t, y = linear_data
    match = MatchingEstimator(prop_score_model=prop_score_model, weighting=weighting)
    match.fit(w, t, y)
    assert match.estimate_ate() == approx(ATE, abs=1)
    lower, upper = match.ate_conf_int()
    assert lower < ATE < upper


@pytest.mark.parametrize('outcome_model', [
    LinearRegression(),
    Lasso(alpha=.1),
    Ridge(alpha=.1),
    ElasticNet(alpha=.01),
    SVR(kernel='linear'),
    LinearSVR(),
    KernelRidge(alpha=.1),
    MLPRegressor(max_iter=1000),
], ids=class_name)
def test_slearner_estimate_near_ate(outcome_model, linear_data):
    w, t, y = linear_data
    slearner = SLearner(outcome_model=outcome_model)
    slearner.fit(w, t, y)
    ate_est = slearner.estimate_ate()
    assert ate_est == approx(ATE, rel=.1)


@pytest.mark.parametrize('outcome_model', [
    LinearRegression(),
    Lasso(alpha=.1),
    Ridge(alpha=.1),
    ElasticNet(alpha=.01),
    SVR(kernel='linear'),
    LinearSVR(),
    KernelRidge(alpha=.1),
], ids=class_name)
def test_slearner_ate_matches_standardization(outcome_model, linear_data):
    w, t, y = linear_data

    slearner = SLearner(outcome_model=outcome_model)
    slearner.fit(w, t, y)
    slearner_ate_est = slearner.estimate_ate()

    standardization = StandardizationEstimator(outcome_model=outcome_model)
    standardization.fit(w, t, y)
    standardization_ate_est = standardization.estimate_ate()

    # NOTE: they are not actually the same
    assert slearner_ate_est == approx(standardization_ate_est, abs=0.3)


@pytest.mark.parametrize('outcome_model', [
    LinearRegression(),
    Lasso(alpha=.1),
    Ridge(alpha=.1),
    ElasticNet(alpha=.01),
    SVR(kernel='linear'),
    LinearSVR(),
    # KernelRidge(alpha=.1),
    MLPRegressor(max_iter=1000),
], ids=class_name)
def test_tlearner_estimate_near_ate(outcome_model, linear_data):
    w, t, y = linear_data
    tlearner = TLearner(outcome_models=outcome_model)
    tlearner.fit(w, t, y)
    ate_est = tlearner.estimate_ate()
    assert ate_est == approx(ATE, rel=.1)


@pytest.mark.parametrize('outcome_model', [
    LinearRegression(),
    Lasso(alpha=.1),
    Ridge(alpha=.1),
    ElasticNet(alpha=.01),
    SVR(kernel='linear'),
    # LinearSVR(),
    KernelRidge(alpha=.1),
], ids=class_name)
def test_tlearner_ate_matches_standardization(outcome_model, linear_data):
    w, t, y = linear_data

    tlearner = TLearner(outcome_models=outcome_model)
    tlearner.fit(w, t, y)
    tlearner_ate_est = tlearner.estimate_ate()

    strat_standard = StratifiedStandardizationEstimator(outcome_models=outcome_model)
    strat_standard.fit(w, t, y)
    strat_standard_ate_est = strat_standard.estimate_ate()

    # NOTE: they ARE actually roughly the same
    assert tlearner_ate_est == approx(strat_standard_ate_est)


@pytest.mark.parametrize('outcome_model', [
    LinearRegression(),
    Lasso(alpha=.1),
    Ridge(alpha=.1),
    ElasticNet(alpha=.01),
    SVR(kernel='linear'),
    LinearSVR(),
    # KernelRidge(alpha=.1),
], ids=class_name)
@pytest.mark.parametrize('prop_score_model', [
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
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
], ids=class_name)
def test_xlearner_estimate_near_ate(outcome_model, prop_score_model, linear_data):
    w, t, y = linear_data
    xlearner = XLearner(outcome_models=outcome_model, prop_score_model=prop_score_model)
    xlearner.fit(w, t, y)
    assert xlearner.estimate_ate() == approx(ATE, rel=.1)


def test_xlearner_different_outcome_models(linear_data):
    w, t, y = linear_data
    xlearner = XLearner(outcome_models=(LinearRegression(), Lasso(alpha=.1)),
                        prop_score_model=LogisticRegression())
    xlearner.fit(w, t, y)
    assert xlearner.estimate_ate() == approx(ATE, rel=.1)


def test_xlearner_different_cate_models(linear_data):
    w, t, y = linear_data
    xlearner = XLearner(cate_models=(LinearRegression(), Lasso(alpha=.1)),
                        prop_score_model=LogisticRegression())
    xlearner.fit(w, t, y)
    assert xlearner.estimate_ate() == approx(ATE, rel=.1)


def test_xlearner_different_outcome_models_and_cate_models(linear_data):
    w, t, y = linear_data
    xlearner = XLearner(outcome_models=(LinearRegression(), Lasso(alpha=.1)),
                        cate_models=(Ridge(alpha=.1), ElasticNet(alpha=.01)),
                        prop_score_model=LogisticRegression())
    xlearner.fit(w, t, y)
    assert xlearner.estimate_ate() == approx(ATE, rel=.1)


@pytest.mark.parametrize('outcome_model', [
    LinearRegression(),
    Lasso(alpha=.1),
    Ridge(alpha=.1),
    ElasticNet(alpha=.01),
    LinearSVR(),
], ids=class_name)
@pytest.mark.parametrize('prop_score_model', [
    LogisticRegression(penalty='l2'),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
], ids=class_name)
@pytest.mark.parametrize('final_model', [
    LinearRegression(),
    Ridge(alpha=.1),
], ids=class_name)
def test_double_ml_estimate_near_ate(outcome_model, prop_score_model, final_model, linear_data):
    w, t, y = linear_data
    double_ml = DoubleML(outcome_model=outcome_model, prop_score_model=prop_score_model, final_model=final_model)
    double_ml.fit(w, t, y)
    assert double_ml.estimate_ate() == approx(ATE, rel=0.5)


@pytest.mark.parametrize('outcome_model', [
    LinearRegression(),
    Lasso(alpha=.1),
    Ridge(alpha=.1),
    ElasticNet(alpha=.01),
    LinearSVR(),
], ids=class_name)
@pytest.mark.parametrize('prop_score_model', [
    LogisticRegression(penalty='l2'),
    # GaussianNB(),
    # QuadraticDiscriminantAnalysis(),
], ids=class_name)
@pytest.mark.parametrize('final_model', [
    LinearRegression(),
    Ridge(alpha=.1),
], ids=class_name)
def test_drlearner_estimate_near_ate(outcome_model, prop_score_model, final_model, linear_data):
    w, t, y = linear_data
    drlearner = DoublyRobustLearner(outcome_model=outcome_model, prop_score_model=prop_score_model, final_model=final_model)
    drlearner.fit(w, t, y)
    assert drlearner.estimate_ate() == approx(ATE, rel=0.6)
