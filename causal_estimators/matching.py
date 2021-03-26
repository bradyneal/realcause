import numpy as np
from scipy import stats
import warnings

import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector
from rpy2.robjects import r
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri; numpy2ri.activate()

from causal_estimators.base import BaseEstimator, NotFittedError
from data.synthetic import generate_wty_linear_multi_w_data

MATCHING = 'Matching'
RGENOUD = 'rgenoud'

INVERSE_VAR = 'inverse_var'
MAHALANOBIS = 'mahalanobis'
GENETIC = 'genetic'
WEIGHT_STR_TO_INDEX = {
    INVERSE_VAR: 1,
    MAHALANOBIS: 2,
    GENETIC: 3,
}


class MatchingEstimator(BaseEstimator):

    def __init__(self, prop_score_model=None, estimand='ATE', weighting=INVERSE_VAR):
        self.use_prop_scores = prop_score_model is not None
        self.prop_score_model = prop_score_model
        self.estimand = estimand
        self.weighting = weighting.lower()
        self.match_out = None
        if self.weighting not in WEIGHT_STR_TO_INDEX.keys():
            raise ValueError('Invalid weighting: {} ... Valid options: {}'
                             .format(weighting, list(WEIGHT_STR_TO_INDEX.keys())))

    def fit(self, w, t, y):
        if self.use_prop_scores:
            self.prop_score_model.fit(w, t)
            match_var = self.prop_score_model.predict(w)
        else:
            match_var = w
        weight_idx = WEIGHT_STR_TO_INDEX[self.weighting]
        weight_matrix = None
        if self.weighting == GENETIC:
            weight_matrix = gen_match(Tr=t, X=match_var)
        self.match_out = match(Y=y, Tr=t, X=match_var, estimand=self.estimand,
                               Weight=weight_idx, Weight_matrix=weight_matrix)

    def estimate_ate(self, t1=1, t0=0, w=None):
        self._error_if_not_fitted()
        return self.match_out.rx2('est')[0]

    def ate_conf_int(self, conf=.95) -> tuple:
        self._error_if_not_fitted()
        if self.use_prop_scores:
            warnings.warn('Confidence interval does not take into account the '
                          'fitting of the propensity score model.')
            # TODO: implement bootstrapping option

        standard_error = self.match_out.rx2('se')[0]
        z_score = stats.norm.ppf((1 + conf) / 2)
        ate_est = self.estimate_ate()
        lower = ate_est - z_score * standard_error
        upper = ate_est + z_score * standard_error
        return lower, upper

    def _error_if_not_fitted(self):
        if self.match_out is None:
            raise NotFittedError()


def match(Y, Tr, X, Z=None, V=None, estimand='ATT', M=1,
          BiasAdjust=False, exact=None, caliper=None, replace=True, ties=True,
          CommonSupport=False, Weight=1, Weight_matrix=None, weights=None,
          Var_calc=0, sample=False, restrict=None, match_out=None,
          distance_tolerance=1e-05, tolerance=None, version="standard"):
    """
    Python interface to the Match function from the Matching R package.
    See R documention here:
    https://www.rdocumentation.org/packages/Matching/versions/4.9-7/topics/Match
    """

    def null_if_none(x):
        if x is None:
            return ro.NULL
        else:
            return x

    # Import (and optionally install) the Matching R package
    if not rpackages.isinstalled(MATCHING):
        utils = importr('utils')
        utils.chooseCRANmirror(ind=1)  # select the first mirror in the list
        utils.install_packages(MATCHING)
    matching = importr(MATCHING)

    # Set argument defaults that were not already set
    if Z is None:
        Z = X
    if V is None:
        V = np.ones(len(Y))
    if tolerance is None:
        tolerance = r('.Machine$double.eps')[0]
    # Y = null_if_none(Y)
    exact = null_if_none(exact)
    caliper = null_if_none(caliper)
    Weight_matrix = null_if_none(Weight_matrix)
    weights = null_if_none(weights)
    restrict = null_if_none(restrict)
    match_out = null_if_none(match_out)

    return matching.Match(Y=Y, Tr=Tr, X=X, Z=Z, V=V, estimand=estimand, M=M,
                          BiasAdjust=BiasAdjust, exact=exact, caliper=caliper,
                          replace=replace, ties=ties, CommonSupport=CommonSupport,
                          Weight=Weight, Weight_matrix=Weight_matrix,
                          weights=weights, Var_calc=Var_calc, sample=sample,
                          restrict=restrict, match_out=match_out,
                          distance_tolerance=distance_tolerance,
                          tolerance=tolerance, version=version)


def gen_match(Tr, X, estimand='ATT'):
    # Install the Matching and rgenoud R packages, if not already installed
    to_install = [x for x in [MATCHING, RGENOUD] if not rpackages.isinstalled(x)]
    if len(to_install) > 0:
        utils = importr('utils')
        utils.chooseCRANmirror(ind=1)  # select the first mirror in the list
        utils.install_packages(StrVector(to_install))
    matching = importr(MATCHING)    # import Matching R package
    importr(RGENOUD)

    return matching.GenMatch(Tr=Tr, X=X, estimand=estimand)


if __name__ == '__main__':
    w, t, y = generate_wty_linear_multi_w_data(100, wdim=5, binary_treatment=True, delta=5)
    m = MatchingEstimator(weighting=GENETIC)
    m.fit(w, t, y)
    est = m.estimate_ate()
    conf_int = m.ate_conf_int()
