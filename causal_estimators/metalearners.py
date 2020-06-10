from sklearn.linear_model import LinearRegression, LogisticRegression
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


class XLearner(BaseEconMLEstimator):

    def __init__(self, outcome_models=LinearRegression(), cate_models=None, prop_score_model=LogisticRegression()):
        """

        :param outcome_models: either a single sklearn-like model or a tuple
            of models, with one for each value of treatment:
            (T=0 model, T=1 model, ...)
        :param cate_models: either a single sklearn-like model or a tuple
            of models, with one for each value of treatment:
            (T=0 model, T=1 model, ...). If None, it will be same models as the
            outcome models.
        :param prop_score_model: An sklearn-like model for the propensity score.
            Must implement fit and predict_proba methods.
        """
        super().__init__(econml.metalearners.XLearner(models=outcome_models,
                                                      cate_models=cate_models,
                                                      propensity_model=prop_score_model))
