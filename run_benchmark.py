from pyro.infer.autoguide import AutoNormal
from DataGenModel import DataGenModel
from data.synthetic import generate_zty_linear_scalar_data, generate_zty_linear_multi_z_data
from data.whynot_simulators import generate_lalonde_random_outcome
from models import linear_gaussian_full_model, linear_gaussian_outcome_model, linear_multi_z_outcome_model, VAE
import numpy as np
import pandas as pd
import whynot as wn
from utils import NUMPY


def run_whynot_estimator_suite(z, t, y, ate):
    # Run the suite of estimates
    estimated_effects = wn.causal_suite(z, t, y)

    # Evaluate the relative error of the estimates
    for estimator, estimate in estimated_effects.items():
        relative_error = np.abs((estimate.ate - ate) / ate)
        print("{}: {:.2f}".format(estimator, relative_error))
    return estimated_effects


def get_whynot_estimator_suite_stats(n_seeds, n_samples=100, ate=2):
    stats = {}
    estimates_data = {}
    for seed in range(n_seeds):
        z, t, y = generate_zty_linear_multi_z_data(n_samples, data_format=NUMPY, binary_treatment=True,
                                                   delta=ate, seed=seed)
        estimated_effects = run_whynot_estimator_suite(z, t, y, ate)
        n_estimators = len(estimated_effects)
        if len(estimates_data) == 0:
            for key, estimate in estimated_effects.items():
                estimates_data[key] = {'ate_ests': np.empty(n_seeds), 'coverate_count': 0}
                stats[key] = {}
        assert estimates_data.keys() == estimated_effects.keys()
        for key, estimate in estimated_effects.items():
            estimates_data[key]['ate_ests'][seed] = estimate.ate
            if estimate.ci[0] < ate < estimate.ci[1]:
                estimates_data[key]['coverate_count'] += 1

    assert stats.keys() == estimates_data.keys()
    for key, estimator_data in estimates_data.items():
        avg_ate_est = np.mean(estimator_data['ate_ests'])
        stats[key]['mse'] = np.mean(np.square(estimator_data['ate_ests'] - ate))
        stats[key]['bias'] = avg_ate_est - ate
        stats[key]['variance'] = np.mean(np.square(estimator_data['ate_ests'] - avg_ate_est))
        stats[key]['coverage_prob'] = estimator_data['coverate_count'] / n_seeds
    return pd.DataFrame.from_dict(stats, orient='index')


if __name__ == '__main__':
    stats = get_whynot_estimator_suite_stats(25)
