from sklearn.linear_model import LinearRegression, LogisticRegression
from econml.dml import NonParamDMLCateEstimator

from causal_estimators.base import BaseEconMLEstimator


class DoubleML(BaseEconMLEstimator):

    def __init__(self, outcome_model=LinearRegression(),
                 prop_score_model=LogisticRegression(),
                 final_model=LinearRegression(), discrete_treatment=True):
        # TODO: add other options that NonParamDMLCateEstimator allows?
        super().__init__(econml_estimator=NonParamDMLCateEstimator(
            model_y=outcome_model, model_t=prop_score_model, model_final=final_model,
            discrete_treatment=discrete_treatment))
