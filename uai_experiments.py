import numpy as np
import pandas as pd
import time

from evaluation import calculate_metrics
from causal_estimators.ipw_estimator import IPWEstimator
from causal_estimators.standardization_estimator import \
    StandardizationEstimator, StratifiedStandardizationEstimator
from evaluation import run_model_cv
from run_metrics import load_from_folder

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
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import cross_val_score, cross_validate


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

outcome_model_grid = [
    ('LinearRegression', LinearRegression(), {}),
    ('LinearRegression_interact',
     make_pipeline(PolynomialFeatures(degree=2, interaction_only=True),
                   LinearRegression()),
     {}),
    ('LinearRegression_degree2',
     make_pipeline(PolynomialFeatures(degree=2), LinearRegression()), {}),
    ('LinearRegression_degree3',
     make_pipeline(PolynomialFeatures(degree=3), LinearRegression()), {}),

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

prop_score_model_grid = [
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
gen_models = [
    ('lalonde_psid', psid_gen_model),
    ('lalonde_cps', cps_gen_model),
    ('twins', twins_gen_model)
]

t_start = time.time()

# TODO: move the loop over seeds out here (rather than in calculuate_metrics and run_model_cv)
# and then use cross_validate() and version of calculuate_metrics() to get scores for all models
# and bundle all the statistical and causal metrics into a single DataFrame with a seed column
# (bias/variance/etc. will need to be calculated from this DataFrame)
# These DataFrames can be used as a dataset for predicting causal performance from predictive performance
N_SEEDS = 5

# for name, outcome_model, param_grid in outcome_model_grid:
dataset_dfs = []
for gen_name, gen_model in gen_models:
    dataset_start = time.time()
    model_dfs = []
    for model_name, outcome_model, param_grid in outcome_model_grid:
        results = run_model_cv(gen_model, outcome_model, model_name=model_name, param_grid=param_grid,
                               n_seeds=N_SEEDS, model_type='outcome', best_model=False, ret_time=True)
        metrics_list = []
        for params in results['params_outcome_model']:
            estimator = StandardizationEstimator(outcome_model=outcome_model.set_params(**params))
            metrics = calculate_metrics(gen_model, estimator, n_seeds=N_SEEDS, conf_ints=False)
            metrics_list.append(metrics)
        causal_metrics = pd.DataFrame(metrics_list)
        model_df = pd.concat([results, causal_metrics], axis=1)
        model_df.insert(0, 'dataset', gen_name)
        model_df.insert(1, 'meta-estimator', 'standardization')
        model_dfs.append(model_df)
    dataset_df = pd.concat(model_dfs, axis=0)
    dataset_end = time.time()
    print(gen_name, 'standardization time:', (dataset_end - dataset_start) / 60 / 60, 'hours')
    dataset_dfs.append(dataset_df)
full_df = pd.concat(dataset_dfs, axis=0)

t_end = time.time()
print('Total time elapsed:', (t_end - t_start) / 60 / 60, 'hours')
full_df.to_csv('results/psid_cps_twins_standardization5.csv', float_format='%.2f', index=False)
