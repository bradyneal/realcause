import numpy as np
import pandas as pd
import time
from pathlib import Path

from experiments.evaluation import calculate_metrics
from causal_estimators.ipw_estimator import IPWEstimator
from causal_estimators.standardization_estimator import \
    StandardizationEstimator, StratifiedStandardizationEstimator
from experiments.evaluation import run_model_cv
from loading import load_from_folder

from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge, ElasticNet, RidgeClassifier
from sklearn.svm import SVR, LinearSVR, SVC, LinearSVC
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor,\
    RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.exceptions import UndefinedMetricWarning

import warnings

warnings.simplefilter(action='ignore', category=UndefinedMetricWarning)
# warnings.filterwarnings("ignore", message="UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.")

RESULTS_DIR = Path('results')

alphas = {'alpha': np.logspace(-4, 5, 10)}
# gammas = [] + ['scale']
Cs = np.logspace(-4, 5, 10)
d_Cs = {'C': Cs}
SVM = 'svm'
d_Cs_pipeline = {SVM + '__C': Cs}
max_depths = list(range(2, 10 + 1)) + [None]
d_max_depths = {'max_depth': max_depths}
d_max_depths_base = {'base_estimator__max_depth': max_depths}
Ks = {'n_neighbors': [1, 2, 3, 5, 10, 15, 25, 50, 100, 200]}

OUTCOME_MODEL_GRID = [
    ('LinearRegression', LinearRegression(), {}),
    ('LinearRegression_interact',
     make_pipeline(PolynomialFeatures(degree=2, interaction_only=True),
                   LinearRegression()),
     {}),
    ('LinearRegression_degree2',
     make_pipeline(PolynomialFeatures(degree=2), LinearRegression()), {}),
    # ('LinearRegression_degree3',
    #  make_pipeline(PolynomialFeatures(degree=3), LinearRegression()), {}),

    ('Ridge', Ridge(), alphas),
    ('Lasso', Lasso(), alphas),
    ('ElasticNet', ElasticNet(), alphas),

    ('KernelRidge', KernelRidge(), alphas),

    ('SVM_rbf', SVR(kernel='rbf'), d_Cs),
    ('SVM_sigmoid', SVR(kernel='sigmoid'), d_Cs),
    ('LinearSVM', LinearSVR(), d_Cs),
    # (SVR(kernel='linear'), d_Cs), # doesn't seem to work (runs forever)

    # TODO: add tuning of SVM gamma, rather than using the default "scale" setting
    # SVMs are sensitive to input scale
    ('Standardized_SVM_rbf', Pipeline([('standard', StandardScaler()), (SVM, SVR(kernel='rbf'))]),
     d_Cs_pipeline),
    ('Standardized_SVM_sigmoid', Pipeline([('standard', StandardScaler()), (SVM, SVR(kernel='sigmoid'))]),
     d_Cs_pipeline),
    ('Standardized_LinearSVM', Pipeline([('standard', StandardScaler()), (SVM, LinearSVR())]),
     d_Cs_pipeline),

    ('kNN', KNeighborsRegressor(), Ks),

    # GaussianProcessRegressor(),

    # TODO: also cross-validate over min_samples_split and min_samples_leaf
    ('DecisionTree', DecisionTreeRegressor(), d_max_depths),
    # ('RandomForest', RandomForestRegressor(), d_max_depths),

    # TODO: also cross-validate over learning_rate
    # ('AdaBoost', AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=None)), d_max_depths_base),
    # ('GradientBoosting', GradientBoostingRegressor(), d_max_depths),

    # MLPRegressor(max_iter=1000),
    # MLPRegressor(alpha=1, max_iter=1000),
]

PROP_SCORE_MODEL_GRID = [
    ('LogisticRegression_l2', LogisticRegression(penalty='l2'), d_Cs),
    ('LogisticRegression', LogisticRegression(penalty='none'), {}),
    ('LogisticRegression_l2_liblinear', LogisticRegression(penalty='l2', solver='liblinear'), d_Cs),
    ('LogisticRegression_l1_liblinear', LogisticRegression(penalty='l1', solver='liblinear'), d_Cs),
    ('LogisticRegression_l1_saga', LogisticRegression(penalty='l1', solver='saga'), d_Cs),

    ('LDA', LinearDiscriminantAnalysis(), {}),
    ('LDA_shrinkage', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'), {}),
    ('QDA', QuadraticDiscriminantAnalysis(), {}),

    # TODO: add tuning of SVM gamma, rather than using the default "scale" setting
    ('SVM_rbf', SVC(kernel='rbf', probability=True), d_Cs),
    ('SVM_sigmoid', SVC(kernel='sigmoid', probability=True), d_Cs),
    # ('SVM_linear', SVC(kernel='linear', probability=True), d_Cs),   # doesn't seem to work (runs forever)

    # SVMs are sensitive to input scale
    ('Standardized_SVM_rbf', Pipeline([('standard', StandardScaler()), (SVM, SVC(kernel='rbf', probability=True))]),
     d_Cs_pipeline),
    ('Standardized_SVM_sigmoid', Pipeline([('standard', StandardScaler()),
                                           (SVM, SVC(kernel='sigmoid', probability=True))]),
     d_Cs_pipeline),
    # ('Standardized_SVM_linear', Pipeline([('standard', StandardScaler()),
    #                                      (SVM, SVC(kernel='linear', probability=True))]),
    #  d_Cs_pipeline),       # doesn't seem to work (runs forever)

    ('kNN', KNeighborsClassifier(), Ks),
    # GaussianProcessClassifier(),

    ('GaussianNB', GaussianNB(), {}),

    # TODO: also cross-validate over min_samples_split and min_samples_leaf
    ('DecisionTree', DecisionTreeClassifier(), d_max_depths),
    # ('RandomForest', RandomForestClassifier(), max_depths),

    # TODO: also cross-validate over learning_rate
    # ('AdaBoost', AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=None)), d_max_depths_base),
    # ('GradientBoosting', GradientBoostingClassifier(), d_max_depths),

    # MLPClassifier(max_iter=1000),
    # MLPClassifier(alpha=1, max_iter=1000),
]

psid_gen_model, args = load_from_folder(dataset='lalonde_psid1')
cps_gen_model, args = load_from_folder(dataset='lalonde_cps1')
twins_gen_model, args = load_from_folder(dataset='twins')

psid_ate = psid_gen_model.ate(noisy=True)
psid_ite = psid_gen_model.ite(noisy=True).squeeze()
cps_ate = cps_gen_model.ate(noisy=True)
cps_ite = cps_gen_model.ite(noisy=True).squeeze()
twins_ate = twins_gen_model.ate(noisy=False)
twins_ite = twins_gen_model.ite(noisy=False).squeeze()

GEN_MODELS = [
    ('lalonde_psid', psid_gen_model, psid_ate, psid_ite),
    ('lalonde_cps', cps_gen_model, cps_ate, cps_ite),
    ('twins', twins_gen_model, twins_ate, twins_ite)
]

t_start = time.time()

N_SEEDS_CV = 5
N_SEEDS_METRICS = 5

def run_experiments_for_estimator(get_estimator_func, model_grid, save_location,
                                  meta_est_name, model_type, exclude=[],
                                  gen_models=GEN_MODELS, n_seeds_cv=N_SEEDS_CV,
                                  n_seeds_metrics=N_SEEDS_METRICS):
    # if outcome_model_grid is None and prop_score_model_grid is None:
    #     raise ValueError('Either outcome_model_grid or prop_score_model_grid must be not None.')
    # if outcome_model_grid is not None and prop_score_model_grid is not None:
    #     raise ValueError('Currently only supporting one non-None model grid.')

    # outcome_modeling = outcome_model_grid is not None
    # model_grid = outcome_model_grid if outcome_modeling else prop_score_model_grid
    # model_type = 'outcome' if outcome_modeling else 'prop_score'
    valid_model_types = ['outcome', 'prop_score']
    if model_type not in valid_model_types:
        raise ValueError('Invalid model_type... Valid model_types: {}'.format(valid_model_types))
    param_str = 'params_' + model_type + '_model'

    dataset_dfs = []
    for gen_name, gen_model, ate, ite in gen_models:
        print('DATASET:', gen_name)
        dataset_start = time.time()
        model_dfs = []
        for model_name, model, param_grid in model_grid:
            print('MODEL:', model_name)
            if (gen_name, model_name) in exclude or model_name in exclude:
                print('SKIPPING')
                continue
            model_start = time.time()
            results = run_model_cv(gen_model, model, model_name=model_name, param_grid=param_grid,
                                   n_seeds=n_seeds_cv, model_type=model_type, best_model=False, ret_time=False)
            metrics_list = []
            for params in results[param_str]:
                try:
                    est_start = time.time()
                    estimator = get_estimator_func(model.set_params(**params))
                    metrics = calculate_metrics(gen_model, estimator, n_seeds=n_seeds_metrics,
                                                conf_ints=False, ate=ate, ite=ite)
                    est_end = time.time()
                    # Add estimator fitting time in minutes
                    metrics['time'] = (est_end - est_start) / 60
                    metrics_list.append(metrics)
                except ValueError:
                    print('Skipping {} params: {}'.format(model_name, params))
            causal_metrics = pd.DataFrame(metrics_list)
            model_df = pd.concat([results, causal_metrics], axis=1)
            model_df.insert(0, 'dataset', gen_name)
            model_df.insert(1, 'meta-estimator', meta_est_name)
            model_dfs.append(model_df)
            model_end = time.time()
            print(model_name, 'time:', (model_end - model_start) / 60, 'minutes')

        dataset_df = pd.concat(model_dfs, axis=0)
        dataset_end = time.time()
        print(gen_name, 'time:', (dataset_end - dataset_start) / 60 / 60, 'hours')
        dataset_dfs.append(dataset_df)
    full_df = pd.concat(dataset_dfs, axis=0)

    t_end = time.time()
    print('Total time elapsed:', (t_end - t_start) / 60 / 60, 'hours')
    full_df.to_csv(save_location, float_format='%.2f', index=False)
    return full_df


print('STANDARDIZATION')
stand_df = run_experiments_for_estimator(
    lambda model: StandardizationEstimator(outcome_model=model),
    model_grid=OUTCOME_MODEL_GRID,
    save_location=RESULTS_DIR / 'psid_cps_twins_standard.csv',
    meta_est_name='standardization',
    model_type='outcome',
    gen_models=GEN_MODELS)

print('STRATIFIED STANDARDIZATION')
strat_df = run_experiments_for_estimator(
    lambda model: StratifiedStandardizationEstimator(outcome_models=model),
    model_grid=OUTCOME_MODEL_GRID,
    exclude=[('lalonde_cps', 'KernelRidge')],
    save_location=RESULTS_DIR / 'psid_cps_twins_strat_standard.csv',
    meta_est_name='stratified_standardization',
    model_type='outcome',
    gen_models=GEN_MODELS)

print('IPW')
ps_df = run_experiments_for_estimator(
    lambda model: IPWEstimator(prop_score_model=model),
    model_grid=PROP_SCORE_MODEL_GRID,
    # exclude=[('lalonde_psid', 'SVM_rbf')],
    exclude=['SVM_rbf'],
    save_location=RESULTS_DIR / 'psid_cps_twins_ipw.csv',
    meta_est_name='ipw',
    model_type='prop_score',
    gen_models=GEN_MODELS)

print('IPW TRIM EPS 0.01')
ps_trim_df = run_experiments_for_estimator(
    lambda model: IPWEstimator(prop_score_model=model, trim_eps=0.01),
    model_grid=PROP_SCORE_MODEL_GRID,
    # exclude=[('lalonde_psid', 'SVM_rbf')],
    exclude=['SVM_rbf'],
    save_location=RESULTS_DIR / 'psid_cps_twins_ipw_trim_01.csv',
    meta_est_name='ipw_trimeps.01',
    model_type='prop_score',
    gen_models=GEN_MODELS)

print('IPW Stabilized weights')
ps_stab_df = run_experiments_for_estimator(
    lambda model: IPWEstimator(prop_score_model=model, stabilized=True),
    model_grid=PROP_SCORE_MODEL_GRID,
    # exclude=[('lalonde_psid', 'SVM_rbf')],
    exclude=['SVM_rbf'],
    save_location=RESULTS_DIR / 'psid_cps_twins_ipw_stabilized.csv',
    meta_est_name='ipw_stabilized',
    model_type='prop_score',
    gen_models=GEN_MODELS)
