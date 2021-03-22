"""
File for loading the Twins semi-synthetic (treatment is simulated) dataset.

Louizos et al. (2017) introduced the Twins dataset as an augmentation of the
real data on twin births and twin mortality rates in the USA from 1989-1991
(Almond et al., 2005). The treatment is "born the heavier twin" so, in one
sense, we can observe both potential outcomes. Louizos et al. (2017) create an
observational dataset out of this by hiding one of the twins (for each pair) in
the dataset. Furthermore, to make sure the twins are very similar, they limit
the data to the twins that are the same sex. To look at data with higher
mortality rates, they further limit the dataset to twins that were born weighing
less than 2 kg. To ensure there is some confounding, Louizos et al. (2017)
simulate the treatment assignment (which twin is heavier) as a function of the
GESTAT10 covariate, which is the number of gestation weeks prior to birth.
GESTAT10 is highly correlated with the outcome and it seems intuitive that it
would be a cause of the outcome, so this should simulate some confounding.

References:

    Almond, D., Chay, K. Y., & Lee, D. S. (2005). The costs of low birth weight.
        The Quarterly Journal of Economics, 120(3), 1031-1083.

    Louizos, C., Shalit, U., Mooij, J. M., Sontag, D., Zemel, R., & Welling, M.
        (2017). Causal effect inference with deep latent-variable models. In
        Advances in Neural Information Processing Systems (pp. 6446-6456).
"""

import os
import pandas as pd
from utils import download_dataset, DATA_FOLDER, NUMPY, PANDAS

TWINS_URL = 'https://raw.githubusercontent.com/shalit-lab/Benchmarks/master/Twins/Final_data_twins.csv'
TWINS_FILENAME = 'twins.csv'


def load_twins(dataroot=DATA_FOLDER, data_format=NUMPY,
               return_sketchy_ites=False, return_sketchy_ate=False,
               observe_sketchy_counterfactuals=False):
    """
    Load the Twins dataset

    :param dataroot: path to folder for data
    :param return_sketchy_ites: if True, return sketchy ITEs
    :param return_sketchy_ate: if True, return sketchy ATE
    :param observe_sketchy_counterfactuals: TODO
    :return: dictionary of results
    """
    if observe_sketchy_counterfactuals:
        raise NotImplementedError('Let Brady know if you need this.')

    download_dataset(TWINS_URL, 'Twins', dataroot=dataroot, filename=TWINS_FILENAME)
    full_df = pd.read_csv(os.path.join(dataroot, TWINS_FILENAME), index_col=0)


    if data_format == NUMPY:
        d = {
            'w': full_df.drop(['T', 'y0', 'y1', 'yf', 'y_cf', 'Propensity'], axis='columns').to_numpy(),
            't': full_df['T'].to_numpy(),
            'y': full_df['yf'].to_numpy()
        }
    elif data_format == PANDAS:
        d = {
            'w': full_df.drop(['T', 'y0', 'y1', 'yf', 'y_cf', 'Propensity'], axis='columns'),
            't': full_df['T'],
            'y': full_df['yf']
        }

    if return_sketchy_ites or return_sketchy_ate:
        ites = full_df['y1'] - full_df['y0']
        ites_np = ites.to_numpy()
        if return_sketchy_ites:
            d['ites'] = ites if data_format == PANDAS else ites_np
        if return_sketchy_ate:
            d['ate'] = ites_np.mean()

    return d
