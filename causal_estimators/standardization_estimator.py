from sklearn.linear_model import LinearRegression
from causallib.estimation import Standardization, StratifiedStandardization

from causal_estimators.base import BaseCausallibIteEstimator


class StandardizationEstimator(BaseCausallibIteEstimator):

    def __init__(self, outcome_model=LinearRegression()):
        super().__init__(causallib_estimator=Standardization(learner=outcome_model))


class StratifiedStandardizationEstimator(BaseCausallibIteEstimator):

    def __init__(self, outcome_model=LinearRegression()):
        super().__init__(causallib_estimator=StratifiedStandardization(learner=outcome_model))
