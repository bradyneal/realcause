from pyro.infer.autoguide import AutoNormal
from DataGenModel import DataGenModel
from data.synthetic import generate_wty_linear_scalar_data, generate_wty_linear_multi_w_data
from data.whynot_simulators import generate_lalonde_random_outcome
from models import linear_gaussian_full_model, linear_gaussian_outcome_model, linear_multi_w_outcome_model, VAE
import numpy as np
import pandas as pd
import whynot as wn
from utils import NUMPY
from functools import partial

import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from plotting import save_and_show

ATE_ESTS = 'ate_ests'
COVERAGE_COUNT = 'coverage_count'
MSE = 'MSE'
BIAS = 'Bias'
ABS_BIAS = '|Bias|'
VARIANCE = 'Variance'
COVERAGE = 'Coverage Probability'
ESTIMATOR = 'Estimators'

NAME = 'benchmark'

class Benchmark:

    def __init__(self, data_generator=None, ate=2, verbose=True):
        if data_generator is None:
            self.data_generator = partial(generate_wty_linear_multi_w_data,
                                          data_format=NUMPY, binary_treatment=True, delta=ate)
        else:
            self.data_generator = data_generator
        self.ate = ate
        self.verbose = verbose
        self.stats = None

    def run_benchmark(self, n_seeds, n_samples=200):
        rename_estimator = {
            'ols': 'OLS',
            'propensity_score_matching': 'Prop Match',
            'propensity_weighted_ols': 'IPW',
            'ip_weighting': 'R IPW',
            'mahalanobis_matching': 'Mahal Match',
            'causal_forest': 'GRF',
            'tmle': 'TMLE',
        }

        stats = {}
        estimates_data = {}
        for seed in range(n_seeds):
            if self.verbose:
                print('Seed:', seed)

            w, t, y = self.data_generator(n_samples, seed=seed)
            estimated_effects = wn.causal_suite(w, t, y, verbose=self.verbose)

            if len(estimates_data) == 0:
                for key, estimate in estimated_effects.items():
                    new_key = rename_estimator[key]
                    estimates_data[new_key] = {ATE_ESTS: np.empty(n_seeds), COVERAGE_COUNT: 0}
                    stats[new_key] = {}
            assert len(estimates_data.keys()) == len(estimated_effects.keys())
            for key, estimate in estimated_effects.items():
                new_key = rename_estimator[key]
                estimates_data[new_key][ATE_ESTS][seed] = estimate.ate
                if estimate.ci[0] < self.ate < estimate.ci[1]:
                    estimates_data[new_key][COVERAGE_COUNT] += 1

        assert stats.keys() == estimates_data.keys()
        for key, estimator_data in estimates_data.items():
            avg_ate_est = np.mean(estimator_data[ATE_ESTS])
            stats[key][MSE] = np.mean(np.square(estimator_data[ATE_ESTS] - self.ate))
            stats[key][BIAS] = avg_ate_est - self.ate
            stats[key][ABS_BIAS] = abs(avg_ate_est - self.ate)
            stats[key][VARIANCE] = np.mean(np.square(estimator_data[ATE_ESTS] - avg_ate_est))
            stats[key][COVERAGE] = estimator_data[COVERAGE_COUNT] / n_seeds

        self.stats = pd.DataFrame.from_dict(stats, orient='index').rename_axis(ESTIMATOR).reset_index()
        return self.stats

    def plot(self, name=NAME, stats=None):
        if stats is None:
            stats = self.stats
        f_mse = plt.figure()
        sns.barplot(x=ESTIMATOR, y=MSE, data=stats.sort_values(MSE, ascending=False), color='royalblue')
        save_and_show(f_mse, '{}_mse.pdf'.format(name))

        f_bias = plt.figure()
        sns.barplot(x=ESTIMATOR, y=ABS_BIAS, data=stats.sort_values(ABS_BIAS, ascending=False), color='royalblue')
        save_and_show(f_bias, '{}_bias.pdf'.format(name))

        f_var = plt.figure()
        sns.barplot(x=ESTIMATOR, y=VARIANCE, data=stats.sort_values(VARIANCE, ascending=False), color='royalblue')
        save_and_show(f_var, '{}_variance.pdf'.format(name))

        f_coverage = plt.figure()
        sns.barplot(x=ESTIMATOR, y=COVERAGE, data=stats.sort_values([COVERAGE, MSE], ascending=[True, False]), color='royalblue')
        save_and_show(f_coverage, '{}_coverage.pdf'.format(name))


if __name__ == '__main__':
    b = Benchmark(ate=2)
    stats = b.run_benchmark(n_seeds=2)
    b.plot(name='benchmark')
    print(stats)
