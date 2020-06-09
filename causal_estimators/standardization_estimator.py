from sklearn.linear_model import LinearRegression
from causallib.estimation import Standardization, StratifiedStandardization

from causal_estimators.base import BaseCausallibIteEstimator


class StandardizationEstimator(BaseCausallibIteEstimator):

    def __init__(self, outcome_model=LinearRegression()):
        super().__init__(causallib_estimator=Standardization(learner=outcome_model))


class StratifiedStandardizationEstimator(BaseCausallibIteEstimator):

    def __init__(self, outcome_models=LinearRegression()):
        """

        :param outcome_models: either a single outcome model to be used for all
            values of treatment, or a dictionary with the treatment values for
            keys and the corresponding outcome model for that treatment as the
            values. Example: {0: LinearRegression(), 1: ElasticNet()}
        """
        super().__init__(causallib_estimator=StratifiedStandardization(learner=outcome_models))
