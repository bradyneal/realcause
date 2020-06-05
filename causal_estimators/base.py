from abc import ABC, abstractmethod
from copy import deepcopy

from utils import to_pandas
from exceptions import NotFittedError


class BaseEstimator(ABC):

    @abstractmethod
    def fit(self, w, t, y):
        pass

    @abstractmethod
    def estimate_ate(self, t1=1, t0=0, w=None):
        pass

    @abstractmethod
    def ate_conf_int(self, percentile=.95) -> tuple:
        pass

    def copy(self):
        return deepcopy(self)


class BaseIteEstimator(BaseEstimator):

    @abstractmethod
    def fit(self, w, t, y):
        pass

    @abstractmethod
    def predict_outcome(self, t, w):
        pass

    def estimate_ate(self, t1=1, t0=0, w=None):
        return self.estimate_ite(t1=t1, t0=t0, w=w).mean()

    @abstractmethod
    def ate_conf_int(self, percentile=.95):
        pass

    @abstractmethod
    def estimate_ite(self, t1=1, t0=0, w=None):
        pass

    # TODO: ITE confidence interval


class BaseCausallibIteEstimator(BaseIteEstimator):

    def __init__(self, causallib_estimator):
        self.causallib_estimator = causallib_estimator
        self.w = None
        self.t = None
        self.y = None

    def fit(self, w, t, y):
        w, t, y = to_pandas(w, t, y)
        self.causallib_estimator.fit(w, t, y)
        self.w = w
        self.t = t
        self.y = y

    def predict_outcome(self, t, w):
        return self.causallib_estimator.estimate_individual_outcome(w, t)

    # def estimate_ate(self, t1=1, t0=0, w=None, t=None, y=None):
    #     w = self.w if w is None else w
    #     t = self.t if t is None else t
    #     y = self.y if y is None else y
    #     if w is None or t is None:
    #         raise RuntimeError('Must run .fit(w, t, y) before running .estimate_ate()')
    #     w, t, y = to_pandas(w, t, y)
    #     mean_potential_outcomes = self.causallib_estimator.estimate_population_outcome(w, t, agg_func="mean")
    #     ate_estimate = mean_potential_outcomes[1] - mean_potential_outcomes[0]
    #     return ate_estimate

    def ate_conf_int(self, percentile=.95):
        # TODO
        raise NotImplementedError

    def estimate_ite(self, t1=1, t0=0, w=None, t=None, y=None):
        w = self.w if w is None else w
        t = self.t if t is None else t
        y = self.y if y is None else y
        if w is None or t is None:
            raise NotFittedError('Must run .fit(w, t, y) before running .estimate_ate()')
        w, t, y = to_pandas(w, t, y)
        individual_potential_outcomes = self.causallib_estimator.estimate_individual_outcome(w, t)
        ite_estimates = individual_potential_outcomes[1] - individual_potential_outcomes[0]
        return ite_estimates
