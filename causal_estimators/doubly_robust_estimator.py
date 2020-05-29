from sklearn.linear_model import LinearRegression, LogisticRegression

from causallib.estimation import Standardization, StratifiedStandardization, IPW,\
    DoublyRobustVanilla, DoublyRobustIpFeature, DoublyRobustJoffe
from causal_estimators.base import BaseIteEstimator
from causal_estimators.standardization_estimator import \
    StandardizationEstimator, StratifiedStandardizationEstimator
from causal_estimators.ipw_estimator import IPWEstimator
from utils import to_pandas

STR_TO_DOUBLY_ROBUST = {
    'vanilla': DoublyRobustVanilla,
    'ipfeature': DoublyRobustIpFeature,
    'joffe': DoublyRobustJoffe,
}
DOUBLY_ROBUST_TYPES = STR_TO_DOUBLY_ROBUST.keys()

STR_TO_STANDARDIZATION = {
    'standardization': Standardization,
    'stratified_standardization': StratifiedStandardization,
    'stratifiedstandardization': StratifiedStandardization,
    'stratified': StratifiedStandardization,
}

TRIM_EPS = 0.01


class DoublyRobustEstimator(BaseIteEstimator):

    def __init__(self, outcome_model=LinearRegression(),
                 prop_score_model=LogisticRegression(),
                 doubly_robust_type='vanilla',
                 standardization_type='standardization',
                 trim_weights=False, trim_eps=None, stabilized=False):

        if doubly_robust_type not in DOUBLY_ROBUST_TYPES:
            raise ValueError('Invalid double_robust_type. Valid types: {}'
                             .format(list(DOUBLY_ROBUST_TYPES)))
        if standardization_type not in STR_TO_STANDARDIZATION.keys():
            raise ValueError('Invalid standardization_type. Valid types: {}'
                             .format(list(STR_TO_STANDARDIZATION.keys())))

        if trim_weights and trim_eps is None:
            trim_eps = TRIM_EPS
        ipw = IPW(learner=prop_score_model, truncate_eps=trim_eps, use_stabilized=stabilized)

        standardization = STR_TO_STANDARDIZATION[standardization_type](outcome_model)
        self.doubly_robust = STR_TO_DOUBLY_ROBUST[doubly_robust_type](
            outcome_model=standardization, weight_model=ipw)
        self.w = None
        self.t = None
        self.y = None

    def fit(self, w, t, y):
        w, t, y = to_pandas(w, t, y)
        self.doubly_robust.fit(w, t, y)
        self.w = w
        self.t = t
        self.y = y

    def predict_outcome(self, w, t):
        return self.doubly_robust.estimate_individual_outcome(w, t)

    # def estimate_ate(self, t1=1, t0=0, w=None, t=None, y=None):
    #     w = self.w if w is None else w
    #     t = self.t if t is None else t
    #     y = self.y if y is None else y
    #     if w is None or t is None:
    #         raise RuntimeError('Must run .fit(w, t, y) before running .estimate_ate()')
    #     w, t, y = to_pandas(w, t, y)
    #     mean_potential_outcomes = self.doubly_robust.estimate_population_outcome(w, t, agg_func="mean")
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
        individual_potential_outcomes = self.doubly_robust.estimate_individual_outcome(w, t)
        ite_estimates = individual_potential_outcomes[1] - individual_potential_outcomes[0]
        return ite_estimates
