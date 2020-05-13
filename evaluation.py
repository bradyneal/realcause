from typing import List
from statistics import mean
from math import sqrt
import numpy as np

from models.base import BaseGenModel
from causal_estimators.base import BaseEstimator

STACK_AXIS = 0
CONF = 0.95


def calculate_metrics(gen_model: BaseGenModel, estimator: BaseEstimator,
                      n_iters: int, conf_ints=True):
    fitted_estimators = []
    for seed in range(n_iters):
        w, t, y = gen_model.sample(seed=seed)
        estimator.fit(w, t, y)
        fitted_estimators.append(estimator.copy())

    ate_metrics = calculate_ate_metrics(gen_model.ate(), fitted_estimators, conf_ints=conf_ints)
    return ate_metrics


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
