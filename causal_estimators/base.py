from abc import ABC, abstractmethod
from copy import deepcopy

from utils import to_pandas


class NotFittedError(ValueError):
    """Exception class to raise if estimator is used before fitting."""

    def __init__(self, msg=None, *args, **kwargs):
        if msg is None:
            msg = 'Call "fit" with appropriate arguments before using this estimator.'
        super().__init__(msg, *args, **kwargs)


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

    def ite_conf_int(self):
        raise NotImplementedError


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
    #         raise NotFittedError('Must run .fit(w, t, y) before running .estimate_ate()')
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
            raise NotFittedError('Must run .fit(w, t, y) before running .estimate_ite()')
        w, t, y = to_pandas(w, t, y)
        individual_potential_outcomes = self.causallib_estimator.estimate_individual_outcome(w, t)
        ite_estimates = individual_potential_outcomes[1] - individual_potential_outcomes[0]
        return ite_estimates


class BaseEconMLEstimator(BaseIteEstimator):

    def __init__(self, econml_estimator):
        self.econml_estimator = econml_estimator
        self.fitted = False
        self.w = None
        self.t = None
        self.y = None

    def fit(self, w, t, y, conf_int_type=None):
        self.econml_estimator.fit(Y=y, T=t, X=w, inference=conf_int_type)
        self.fitted = True
        self.w = w
        self.t = t
        self.y = y

    def predict_outcome(self, t, w):
        raise NotImplementedError

    def ate_conf_int(self, t1=1, t0=0, w=None, percentile=.95):
        raise NotImplementedError

    def estimate_ite(self, t1=1, t0=0, w=None):
        w = self.w if w is None else w
        self._raise_exception_if_not_fitted()
        return self.econml_estimator.effect(T0=t0, T1=t1, X=w)

    def ite_conf_int(self, t1=1, t0=0, w=None, percentile=.95):
        w = self.w if w is None else w
        self._raise_exception_if_not_fitted()
        return self.econml_estimator.effect_interval(T0=t0, T1=t1, X=w, alpha=(1 - percentile))

    def _raise_exception_if_not_fitted(self):
        if not self.fitted:
            raise NotFittedError('Must run .fit(w, t, y) before running .estimate_ite()')
