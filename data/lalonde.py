"""
File for loading LaLonde dataset. The original data comes from Lalonde (1986),
but the data from Dehejia & Wahba (1999) is more commonly used. Dehejia & Wahba (2002)
and Smith and Todd (2005) both use the subset of the LaLonde RCT data from
Dehejia & Wahba (1999), and use the observational subsets PSID-1 and CPS-1.
Angrist & Pischke (2008) use the CPS-1 and CPS-3 data, criticising the CPS-3
subset as being selected in too ad-hoc a fashion. Firpo (2007) only used PSID-1
data for the control group. There seems to be a preference in the literature for
PSID-1 and CPS-1 over their subsetted counterparts (PSID-2, PSID-3, CPS-2, and
CPS-3). However, there is no clear preference between PSID-1 and CPS-1. For the
RCT data, there is clear preference for Dehejia & Wahba (1999)'s subset that
includes the additional covariate (earnings in 1974) over the RCT data from the
original LaLonde (1986) paper.

We recommend to use BOTH the Dehejia & Wahba (1999) RCT data with PSID-1 control
and the Dehejia & Wahba (1999) RCT data with CPS-1 control. We use
Dehejia & Wahba (1999)'s RCT data and PSID-1 control as default.


Data source: http://users.nber.org/~rdehejia/nswdata2.html


References:

Angrist, J. D. and Pischke, J.-S. (2008). Mostly Harmless Econometrics: An Empiricist's Companion.
    Princeton University Press.

Dehejia, R.H. and Wahba, S. (1999). Causal Effects in Nonexperimental Studies:
    Re-Evaluating the Evaluation of Training Programs. Journal of the American Statistical Association 94: 1053-1062.

Dehejia, R.H. and Wahba, S. (2002). Propensity Score Matching Methods for Non-Experimental Causal Studies.
    Review of Economics and Statistics, Vol. 84, (February 2002), pp. 151-161.

Sergio Firpo. (2007). Efficient Semiparametric Estimation of Quantile Treatment Effects.
    Econometrica, 75(1), 259-276.

Lalonde, R. (1986). Evaluating the econometric evaluations of training programs with experimental data.
    American Economic Review 76: 604-620.

Smith, J. A. and Todd, P. E. (2005). Does matching overcome LaLonde's critique of nonexperimental estimators?
    Journal of Econometrics, Elsevier, vol. 125(1-2), pages 305-353.
"""

import os
import pandas as pd
from utils import to_data_format, NUMPY, PANDAS_SINGLE, DATA_FOLDER


DEHEJIA_WAHBA = 'dw'
LALONDE = 'lalonde'
PSID = 'psid1'


def load_lalonde(rct_version=DEHEJIA_WAHBA, obs_version=PSID, rct=False, data_format=NUMPY, dataroot=None):
    """
    Load LaLonde dataset: RCT or combined RCT with observational control group
    Options for 2 x 6 = 12 different observational datasets and 2 RCT datasets

    :param rct_version: 'lalonde' for LaLonde (1986)'s original RCT data or 'dw' for Dehejia & Wahba (1999)'s RCT data
    :param obs_version: observational data to use for the control group
    :param rct: use RCT data for both the treatment AND control groups (no observational data)
    :param data_format: returned data format: 'torch' Tensors, 'pandas' DataFrame, or 'numpy' ndarrays
    :return: (covariates, treatment, outcome) tuple or Pandas DataFrame
    """
    rct_df = load_lalonde_rct(rct_version, dataroot=dataroot)
    if rct:
        df = rct_df
    else:
        obs_df = load_lalonde_obs(obs_version, dataroot=dataroot)
        if rct_version == LALONDE:
            # original lalonde dataset doesn't have 1974 earnings
            obs_df.drop('re74', axis='columns', inplace=True)
        # Replace RCT control group with observational data
        combined_df = rct_df[rct_df.treat == 1].append(obs_df)
        df = combined_df
    if data_format.lower() == PANDAS_SINGLE:
        return df
    else:
        w = df.drop(['data_id', 'treat', 're78'], axis='columns')
        t = df['treat']
        y = df['re78']
        return to_data_format(data_format, w, t, y)


def load_lalonde_rct(version=DEHEJIA_WAHBA, dataroot=None):
    if dataroot is None:
        dataroot = DATA_FOLDER
    rct_version_to_name = {
        DEHEJIA_WAHBA: 'nsw_dw.dta',
        LALONDE: 'nsw.dta'
    }
    version = version.lower()
    if version not in rct_version_to_name.keys():
        raise ValueError('Invalid version {} ... Valid versions: {}'.format(version, rct_version_to_name.keys()))
    else:
        return pd.read_stata(os.path.join(dataroot, rct_version_to_name[version]))


def load_lalonde_obs(version=PSID, dataroot=None):
    if dataroot is None:
        dataroot = DATA_FOLDER
    obs_version_to_name = {
        'psid': 'psid_controls.dta',
        'psid1': 'psid_controls.dta',
        'psid2': 'psid_controls2.dta',
        'psid3': 'psid_controls3.dta',
        'cps': 'cps_controls.dta',
        'cps1': 'cps_controls.dta',
        'cps2': 'cps_controls2.dta',
        'cps3': 'cps_controls3.dta',
    }
    version = version.lower()
    if version not in obs_version_to_name.keys():
        raise ValueError('Invalid version {} ... Valid versions: {}'.format(version, obs_version_to_name.keys()))
    else:
        return pd.read_stata(os.path.join(dataroot, obs_version_to_name[version]))
