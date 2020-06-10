import numpy as np
import pandas as pd
import time

from data.lalonde import load_lalonde
from models.linear import LinearGenModel
from models.nonlinear import MLP, TrainingParams, MLPParams, Scaling, Standardize
from evaluation import calculate_metrics
from causal_estimators.ipw_estimator import IPWEstimator
from causal_estimators.standardization_estimator import \
    StandardizationEstimator, StratifiedStandardizationEstimator
from evaluation import run_model_cv

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

from data.synthetic import generate_wty_linear_multi_w_data

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

# w, t, y = generate_wty_linear_multi_w_data(100, binary_treatment=True)
# gen_model = LinearGenModel(w, t, y, binary_treatment=True)

w, t, y = load_lalonde()
gen_model = MLP(w, t, y,
                training_params=TrainingParams(lr=0.0005, batch_size=128, num_epochs=500),
                mlp_params_y_tw=MLPParams(n_hidden_layers=2, dim_h=256),
                binary_treatment=True, outcome_distribution='mixed_log_logistic',
                outcome_min=0.0, outcome_max=1.0, seed=1,
                w_transform=Standardize(w), y_transform=Scaling(1 / y.max()))

t_start = time.time()

# TODO: move the loop over seeds out here (rather than in calculuate_metrics and run_model_cv)
# and then use cross_validate() and version of calculuate_metrics() to get scores for all models
# and bundle all the statistical and causal metrics into a single DataFrame with a seed column
# (bias/variance/etc. will need to be calculated from this DataFrame)
# These DataFrames can be used as a dataset for predicting causal performance from predictive performance
N_SEEDS = 5
metrics_list = []
for name, outcome_model, param_grid in outcome_model_grid:
    results = run_model_cv(gen_model, outcome_model, param_grid=param_grid, n_seeds=N_SEEDS, model_type='outcome',
                           best_model=True)
    estimator = StandardizationEstimator(outcome_model=results['best_model'])
    metrics = calculate_metrics(gen_model, estimator, n_seeds=N_SEEDS, conf_ints=False)
    metrics_list.append({'name': name, **metrics})
    # print(type(outcome_model))
    # print(metrics)

    res = results['df'][['params', 'mean_test_neg_root_mean_squared_error', 'mean_test_r2', 'rank_test_r2', 'seed']]
    print('CV results:')
    print(res)

    print('single ATE estimate:', estimator.estimate_ate())
    bias_ate = gen_model.ate()
    print('Bias ATE:', bias_ate)
    print('single estimate BIAS:', estimator.estimate_ate() - bias_ate)

metrics_df = pd.DataFrame(metrics_list)[['ate_bias', 'ate_abs_bias', 'ate_std_error', 'ate_rmse', 'mean_pehe']]
print(metrics_df)
standardization_df = pd.DataFrame(metrics_list)


metrics_list = []
for name, outcome_model, param_grid in outcome_model_grid:
    results = run_model_cv(gen_model, outcome_model, param_grid=param_grid, n_seeds=N_SEEDS, model_type='outcome',
                           best_model=True)
    estimator = StratifiedStandardizationEstimator(outcome_models=results['best_model'])
    metrics = calculate_metrics(gen_model, estimator, n_seeds=N_SEEDS, conf_ints=False)
    metrics_list.append({'name': name, **metrics})
    print(type(outcome_model))
    print(metrics)
stratified_standardization_df = pd.DataFrame(metrics_list)

metrics_list = []
for name, prop_score_model, param_grid in prop_score_model_grid:
    results = run_model_cv(gen_model, prop_score_model, param_grid=param_grid, n_seeds=N_SEEDS, model_type='prop_score',
                           best_model=True)

    estimator = IPWEstimator(prop_score_model=results['best_model'])
    metrics_list.append({'name': name,
                         **calculate_metrics(gen_model, estimator, n_seeds=N_SEEDS, conf_ints=False)})

    estimator = IPWEstimator(prop_score_model=results['best_model'], trim_weights=True)
    metrics_list.append({'name': name + '_trim_weights',
                         **calculate_metrics(gen_model, estimator, n_seeds=N_SEEDS, conf_ints=False)})

    estimator = IPWEstimator(prop_score_model=results['best_model'], stabilized=True)
    metrics_list.append({'name': name + '_stabilized_weights',
                         **calculate_metrics(gen_model, estimator, n_seeds=N_SEEDS, conf_ints=False)})

    estimator = IPWEstimator(prop_score_model=results['best_model'], trim_weights=True, stabilized=True)
    metrics_list.append({'name': name + '_trim_stabilized_weights',
                         **calculate_metrics(gen_model, estimator, n_seeds=N_SEEDS, conf_ints=False)})
    print('{}:'.format(name))
    print(metrics)
ipw_df = pd.DataFrame(metrics_list)

t_end = time.time()
t_elapsed = t_end - t_start
print('seconds elapsed:', t_elapsed)

# Save dataframes
standardization_df[['name', 'ate_bias', 'ate_abs_bias', 'ate_std_error', 'ate_rmse', 'mean_pehe']].to_csv('results/standardization100.csv', float_format='%.2f', index=False)
stratified_standardization_df[['name', 'ate_bias', 'ate_abs_bias', 'ate_std_error', 'ate_rmse', 'mean_pehe']].to_csv('results/stratified_standardization100.csv', float_format='%.2f', index=False)
ipw_df[['name', 'ate_bias', 'ate_abs_bias', 'ate_std_error', 'ate_rmse']].to_csv('results/ipw100.csv', float_format='%.2f', index=False)
