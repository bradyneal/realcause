from sklearn.linear_model import LinearRegression

from causallib.estimation import Standardization, StratifiedStandardization
from causal_estimators.base import BaseIteEstimator
from utils import to_pandas


class StandardizationEstimator(BaseIteEstimator):

    def __init__(self, outcome_model=LinearRegression()):
        self.standardization = Standardization(learner=outcome_model)
        self.w = None
        self.t = None
        self.y = None

    def fit(self, w, t, y):
        w, t, y = to_pandas(w, t, y)
        self.standardization.fit(w, t, y)
        self.w = w
        self.t = t
        self.y = y

    def predict_outcome(self, w, t):
        return self.standardization.estimate_individual_outcome(w, t)

    # def estimate_ate(self, t1=1, t0=0, w=None, t=None, y=None):
    #     w = self.w if w is None else w
    #     t = self.t if t is None else t
    #     y = self.y if y is None else y
    #     if w is None or t is None:
    #         raise RuntimeError('Must run .fit(w, t, y) before running .estimate_ate()')
    #     w, t, y = to_pandas(w, t, y)
    #     mean_potential_outcomes = self.standardization.estimate_population_outcome(w, t, agg_func="mean")
    #     ate_estimate = mean_potential_outcomes[1] - mean_potential_outcomes[0]
    #     return ate_estimate

    def ate_conf_int(self, percentile=.95) -> tuple:
        raise NotImplementedError

    def estimate_ite(self, t1=1, t0=0, w=None, t=None, y=None):
        w = self.w if w is None else w
        t = self.t if t is None else t
        y = self.y if y is None else y
        if w is None or t is None:
            raise RuntimeError('Must run .fit(w, t, y) before running .estimate_ate()')
        w, t, y = to_pandas(w, t, y)
        individual_potential_outcomes = self.standardization.estimate_individual_outcome(w, t)
        ite_estimates = individual_potential_outcomes[1] - individual_potential_outcomes[0]
        return ite_estimates


class StratifiedStandardizationEstimator(BaseIteEstimator):

    def __init__(self, outcome_model=LinearRegression()):
        self.stratified_standardization = StratifiedStandardization(learner=outcome_model)
        self.w = None
        self.t = None
        self.y = None

    def fit(self, w, t, y):
        w, t, y = to_pandas(w, t, y)
        self.stratified_standardization.fit(w, t, y)
        self.w = w
        self.t = t
        self.y = y

    def predict_outcome(self, w, t):
        return self.stratified_standardization.estimate_individual_outcome(w, t)

    # def estimate_ate(self, t1=1, t0=0, w=None, t=None, y=None):
    #     w = self.w if w is None else w
    #     t = self.t if t is None else t
    #     y = self.y if y is None else y
    #     if w is None or t is None:
    #         raise RuntimeError('Must run .fit(w, t, y) before running .estimate_ate()')
    #     w, t, y = to_pandas(w, t, y)
    #     mean_potential_outcomes = self.stratified_standardization.estimate_population_outcome(w, t, agg_func="mean")
    #     ate_estimate = mean_potential_outcomes[1] - mean_potential_outcomes[0]
    #     return ate_estimate

    def ate_conf_int(self, percentile=.95) -> tuple:
        raise NotImplementedError

    def estimate_ite(self, t1=1, t0=0, w=None, t=None, y=None):
        w = self.w if w is None else w
        t = self.t if t is None else t
        y = self.y if y is None else y
        if w is None or t is None:
            raise RuntimeError('Must run .fit(w, t, y) before running .estimate_ate()')
        w, t, y = to_pandas(w, t, y)
        individual_potential_outcomes = self.stratified_standardization.estimate_individual_outcome(w, t)
        ite_estimates = individual_potential_outcomes[1] - individual_potential_outcomes[0]
        return ite_estimates
