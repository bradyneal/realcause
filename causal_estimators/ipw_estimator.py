from sklearn.linear_model import LogisticRegression
from causallib.estimation import IPW

from causal_estimators.base import BaseEstimator
from utils import to_pandas

TRIM_EPS = 0.01


class IPWEstimator(BaseEstimator):

    def __init__(self, prop_score_model=LogisticRegression(), trim_weights=False, trim_eps=None, stabilized=False):
        if trim_weights and trim_eps is None:
            trim_eps = TRIM_EPS
        self.ipw = IPW(learner=prop_score_model, truncate_eps=trim_eps, use_stabilized=stabilized)
        self.w = None
        self.t = None
        self.y = None

    def fit(self, w, t, y):
        w, t, y = to_pandas(w, t, y)
        self.ipw.fit(w, t)
        self.w = w
        self.t = t
        self.y = y

    def estimate_ate(self, t1=1, t0=0, w=None, t=None, y=None):
        w = self.w if w is None else w
        t = self.t if t is None else t
        y = self.y if y is None else y
        if w is None or t is None or y is None:
            raise RuntimeError('Must run .fit(w, t, y) before running .estimate_ate()')
        w, t, y = to_pandas(w, t, y)
        mean_potential_outcomes = self.ipw.estimate_population_outcome(w, t, y, treatment_values=[t0, t1])
        ate_estimate = mean_potential_outcomes[1] - mean_potential_outcomes[0]
        # Use below estimate_effect() method if want to allow for effects that are not differences
        # ate_estimate = self.ipw.estimate_effect(mean_potential_outcomes[1], mean_potential_outcomes[0])[0]
        return ate_estimate

    def ate_conf_int(self, percentile=.95) -> tuple:
        raise NotImplementedError
