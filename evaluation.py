from typing import List
from statistics import mean
from math import sqrt
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.model_selection._search import BaseSearchCV

from models.base import BaseGenModel
from causal_estimators.base import BaseEstimator, BaseIteEstimator

STACK_AXIS = 0
CONF = 0.95
REGRESSION_SCORES = ['max_error', 'neg_mean_absolute_error', 'neg_median_absolute_error',
                     'neg_mean_squared_error', 'neg_root_mean_squared_error', 'r2']
REGRESSION_SCORE_DEF = 'r2'
CLASSIFICATION_SCORES = ['accuracy', 'balanced_accuracy', 'average_precision',
                         'f1',
                         'precision',
                         'recall', 'roc_auc']
CLASSIFICATION_SCORE_DEF = 'accuracy'


def run_model_cv(gen_model: BaseGenModel, model: sklearn.base.BaseEstimator, model_name: str,
                 param_grid, n_seeds: int, model_type: str, n_folds=5,
                 scoring=None, rank_score=None, best_params=False, best_model=False,
                 ret_time=False):
    model_type = model_type.lower()
    if model_type == 'outcome':
        if scoring is None:
            scoring = REGRESSION_SCORES
        if rank_score is None:
            rank_score = REGRESSION_SCORE_DEF
    elif model_type == 'prop_score' or model_type == 'ps':
        if scoring is None:
            scoring = CLASSIFICATION_SCORES
        if rank_score is None:
            rank_score = CLASSIFICATION_SCORE_DEF
    else:
        raise ValueError('Invalid model_type: {}'.format(model_type))
    cv = GridSearchCV(model, param_grid, scoring=scoring, cv=n_folds, refit=rank_score)
    dfs = []
    for seed in range(n_seeds):
        w, t, y = gen_model.sample(seed=seed)
        if model_type == 'outcome':
            if t.ndim == 1:
                t = t[:, np.newaxis]
            elif t.ndim == 2:
                pass
            else:
                raise ValueError('Invalid dimension of t: {}'.format(t.ndim))
            y = y.squeeze()
            X = np.hstack((w, t))
            cv.fit(X, y)
        else:   # propensity score model_type
            t = t.squeeze()
            cv.fit(w, t)
        d = {k: v for k, v in cv.cv_results_.items() if 'split' not in k}
        dfs.append(pd.DataFrame(d))

    df = pd.concat(dfs, axis=0)
    params = df.params
    cols = [col for col, is_param in zip(df.columns, df.columns.str.startswith('param_')) if is_param]

    # mean over seeds
    if len(cols) > 0:
        agg_df = df.drop('params', axis='columns').groupby(cols).mean().reset_index()
    else:
        agg_df = pd.DataFrame(df.drop('params', axis='columns').mean()).transpose()
    agg_df.insert(0, model_type + '_model', model_name)
    agg_df.insert(1, 'params_' + model_type + '_model', pd.Series(params.iloc[:len(agg_df)], index=agg_df.index))

    # remove timing columns
    if not ret_time:
        time_cols = [col for col, is_time in zip(df.columns, df.columns.str.endswith('time')) if is_time]
        agg_df.drop(time_cols, axis='columns', inplace=True)

    if best_params or best_model:
        results = {'df': agg_df}
        rank_col = 'rank_test_{}'.format(rank_score)
        best_cv = agg_df[agg_df[rank_col] == agg_df[rank_col].min()]
        best_cv_params = best_cv.params.iloc[0]
        best_cv_model = model.set_params(**best_cv_params)
        if best_params:
            results['best_params'] = best_cv_model.get_params()
        if best_model:
            results['best_model'] = best_cv_model
        return results
    else:
        return agg_df


def calculate_outcome_model_scores(gen_model: BaseGenModel, estimator: sklearn.base.BaseEstimator,
                                   n_seeds: int, n_folds=5, ret='mean'):

    def result_key(score):
        return '{}fold_{}'.format(n_folds, score)

    results = {result_key(score): np.zeros(n_seeds) for score in REGRESSION_SCORES}
    for seed in range(n_seeds):
        w, t, y = gen_model.sample(seed=seed)
        if t.ndim == 1:
            t = t[:, np.newaxis]
        elif t.ndim == 2:
            pass
        else:
            raise ValueError('Invalid dimension of t {}'.format(t.ndim))
        X = np.hstack((w, t))
        cv_results = cross_validate(estimator, X, y, cv=n_folds, scoring=scores)
        for score in REGRESSION_SCORES:
            results[result_key(score)][seed] = cv_results['test_' + score].mean()

    ret = ret.lower()
    if ret == 'mean':
        mean_results = {'mean_{}'.format(k): v.mean() for k, v in results.items()}
        return mean_results
    elif ret == 'all' or 'seeds' in ret:
        return results


def calculate_metrics(gen_model: BaseGenModel, estimator: BaseEstimator,
                      n_seeds: int, conf_ints=True, return_ite_vectors=False,
                      ate=None, ite=None):
    if ate is None:
        ate = gen_model.ate()
    # else:
    #     print('Already computed ate')
    if ite is None:
        ite = gen_model.ite().squeeze()
    # else:
    #     print('Already computed ite')
    fitted_estimators = []
    for seed in range(n_seeds):
        w, t, y = gen_model.sample(seed=seed)
        estimator.fit(w, t, y)
        fitted_estimators.append(estimator.copy())

    ate_metrics = calculate_ate_metrics(ate, fitted_estimators, conf_ints=conf_ints)

    is_ite_estimator = isinstance(estimator, BaseIteEstimator)
    if is_ite_estimator:
        ite_metrics = calculate_ite_metrics(ite, fitted_estimators)
        ite_mean_metrics = {'mean_' + k: np.mean(v) for k, v in ite_metrics.items()}
        ite_std_metrics = {'std_of_' + k: np.std(v) for k, v in ite_metrics.items()}

    metrics = ate_metrics
    if is_ite_estimator:
        metrics.update(ite_mean_metrics)
        metrics.update(ite_std_metrics)
        if return_ite_vectors:
            metrics.update(ite_metrics)
    return metrics


def calculate_ate_metrics(ate: float, fitted_estimators: List[BaseEstimator], conf_ints=True):
    ate_estimates = [fitted_estimator.estimate_ate() for
                     fitted_estimator in fitted_estimators]
    mean_ate_estimate = mean(ate_estimates)
    ate_mean_abs_error = mean(abs(ate_estimate - ate) for ate_estimate in ate_estimates)
    ate_bias = mean_ate_estimate - ate
    ate_abs_bias = abs(ate_bias)
    ate_squared_bias = ate_bias**2
    ate_variance = calc_variance(ate_estimates, mean_ate_estimate)
    ate_std_error = sqrt(ate_variance)
    ate_mse = calc_mse(ate_estimates, ate)
    ate_rmse = sqrt(ate_mse)
    metrics = {
        'ate_mean_abs_error': ate_mean_abs_error,
        'ate_bias': ate_bias,
        'ate_abs_bias': ate_abs_bias,
        'ate_squared_bias': ate_squared_bias,
        'ate_variance': ate_variance,
        'ate_std_error': ate_std_error,
        'ate_mse': ate_mse,
        'ate_rmse': ate_rmse,
    }

    if conf_ints:
        ate_conf_ints = [fitted_estimator.ate_conf_int(CONF) for
                         fitted_estimator in fitted_estimators]
        metrics['ate_coverage'] = calc_coverage(ate_conf_ints, ate)
        metrics['ate_mean_int_length']: calc_mean_interval_length(ate_conf_ints)

    return metrics


def calculate_ite_metrics(ite: np.ndarray, fitted_estimators: List[BaseIteEstimator]):
    ite_estimates = np.stack([fitted_estimator.estimate_ite() for
                              fitted_estimator in fitted_estimators],
                             axis=STACK_AXIS)

    # Calculated for each unit/individual, this is the a vector of num units
    mean_ite_estimate = ite_estimates.mean(axis=STACK_AXIS)
    ite_bias = mean_ite_estimate - ite
    ite_abs_bias = np.abs(ite_bias)
    ite_squared_bias = ite_bias**2
    ite_variance = calc_vector_variance(ite_estimates, mean_ite_estimate)
    ite_std_error = np.sqrt(ite_variance)
    ite_mse = calc_vector_mse(ite_estimates, ite)
    ite_rmse = np.sqrt(ite_mse)

    # Calculated for a single dataset, so this is a vector of num datasets
    pehe_squared = calc_vector_mse(ite_estimates, ite, reduce_axis=(1 - STACK_AXIS))
    pehe = np.sqrt(pehe_squared)

    # TODO: ITE coverage
    # ate_coverage = calc_coverage(ate_conf_ints, ate)
    # ate_mean_int_length = calc_mean_interval_length(ate_conf_ints)

    return {
        'ite_bias': ite_bias,
        'ite_abs_bias': ite_abs_bias,
        'ite_squared_bias': ite_squared_bias,
        'ite_variance': ite_variance,
        'ite_std_error': ite_std_error,
        'ite_mse': ite_mse,
        'ite_rmse': ite_rmse,
        # 'ite_coverage': ite_coverage,
        # 'ite_mean_int_length': ite_mean_int_length,
        'pehe_squared': pehe_squared,
        'pehe': pehe,
    }


def calc_variance(estimates, mean_estimate):
    return calc_mse(estimates, mean_estimate)


def calc_mse(estimates, target):
    if isinstance(estimates, (list, tuple)):
        estimates = np.array(estimates)
    return ((estimates - target) ** 2).mean()


def calc_coverage(intervals: List[tuple], estimand):
    n_covers = sum(1 for interval in intervals if interval[0] <= estimand <= interval[1])
    return n_covers / len(intervals)


def calc_mean_interval_length(intervals: List[tuple]):
    return mean(interval[1] - interval[0] for interval in intervals)


def calc_vector_variance(estimates: np.ndarray, mean_estimate: np.ndarray):
    return calc_vector_mse(estimates, mean_estimate)


def calc_vector_mse(estimates: np.ndarray, target: np.ndarray, reduce_axis=STACK_AXIS):
    assert isinstance(estimates, np.ndarray) and estimates.ndim == 2
    assert isinstance(target, np.ndarray) and target.ndim == 1
    assert target.shape[0] == estimates.shape[1 - STACK_AXIS]

    n_seeds = estimates.shape[STACK_AXIS]
    target = np.expand_dims(target, axis=STACK_AXIS).repeat(n_seeds, axis=STACK_AXIS)
    return ((estimates - target) ** 2).mean(axis=reduce_axis)
