from sklearn.linear_model import LinearRegression
import econml.metalearners

from causal_estimators.base import BaseEconMLEstimator


class SLearner(BaseEconMLEstimator):

    def __init__(self, outcome_model=LinearRegression()):
        super().__init__(econml.metalearners.SLearner(outcome_model))


class TLearner(BaseEconMLEstimator):

    def __init__(self, outcome_models=LinearRegression()):
        """

        :param outcome_models: either a single sklearn-like model or a tuple
            of models, with one for each value of treatment:
            (T=0 model, T=1 model, ...)
        """
        super().__init__(econml.metalearners.TLearner(outcome_models))
