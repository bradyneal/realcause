from typing import List
from statistics import mean
from math import sqrt
import numpy as np

from models.base import BaseGenModel
from causal_estimators.base import BaseEstimator, BaseIteEstimator

STACK_AXIS = 0
CONF = 0.95


def calculate_metrics(gen_model: BaseGenModel, estimator: BaseEstimator,
                      n_iters: int, conf_ints=True, return_ite_vectors=False):
    fitted_estimators = []
    for seed in range(n_iters):
        w, t, y = gen_model.sample(seed=seed)
        estimator.fit(w, t, y)
        fitted_estimators.append(estimator.copy())

    ate_metrics = calculate_ate_metrics(gen_model.ate(), fitted_estimators, conf_ints=conf_ints)

    is_ite_estimator = isinstance(estimator, BaseIteEstimator)
    if is_ite_estimator:
        ite_metrics = calculate_ite_metrics(gen_model.ite().squeeze(), fitted_estimators)
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
    ate_bias = mean_ate_estimate - ate
    ate_abs_bias = abs(ate_bias)
    ate_squared_bias = ate_bias**2
    ate_variance = calc_variance(ate_estimates, mean_ate_estimate)
    ate_std_error = sqrt(ate_variance)
    ate_mse = calc_mse(ate_estimates, ate)
    ate_rmse = sqrt(ate_mse)
    metrics = {
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

    # Calulcated for each unit/individual, this is the a vector of num units
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

    n_iters = estimates.shape[STACK_AXIS]
    target = np.expand_dims(target, axis=STACK_AXIS).repeat(n_iters, axis=STACK_AXIS)
    return ((estimates - target) ** 2).mean(axis=reduce_axis)
