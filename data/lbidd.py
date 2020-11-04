"""
File for loading the LBIDD semi-synthetic dataset.

Shimoni et al. (2018) took the real covariates from the Linked Births and Infant
Deaths Database (lbidd) (MacDorman & Atkinson, 1998) and generated
semi-synthetic data by generating the treatment assignments and outcomes via
random functions of the covariates.

Data Wiki: https://www.synapse.org/#!Synapse:syn11738767/wiki/512854
CDC Data Website: https://www.cdc.gov/nchs/nvss/linked-birth.htm

References:

    MacDorman, Marian F and Jonnae O Atkinson. Infant mortality statistics from
        the linked birth/infant death data set—1995 period data. Mon Vital Stat
        Rep, 46(suppl 2):1–22, 1998.
        https://www.cdc.gov/nchs/data/mvsr/supp/mv46_06s2.pdf

    Shimoni, Y., Yanover, C., Karavani, E., & Goldschmidt, Y. (2018).
        Benchmarking Framework for Performance-Evaluation of Causal Inference
        Analysis. ArXiv, abs/1802.05046.
"""

import os
import numpy as np
import pandas as pd
import tarfile

from utils import download_dataset, unzip, DATA_FOLDER

LBIDD_FOLDER = 'lbidd'
SCALING_FOLDER = 'scaling'
SCALING_TAR_ZIP = 'scaling.tar.gz'
FILE_EXT = '.csv'
INDEX_COL_NAME = 'sample_id'
COUNTERFACTUAL_FILE_SUFFIX = '_cf'

def get_paths(dataroot=None):
    if dataroot is None:
        dataroot = DATA_FOLDER
    folder = os.path.join(dataroot, 'lbidd')
    scaling_zip = os.path.join(folder, 'scaling.tar.gz')
    scaling_folder = os.path.join(folder, 'scaling')
    covariates_path = os.path.join(folder, 'x.csv')
    params_path = os.path.join(scaling_folder, 'params.csv')
    counterfactuals_folder = os.path.join(scaling_folder, 'counterfactuals')
    factuals_folder = os.path.join(scaling_folder, 'factuals')
    return folder, scaling_zip, scaling_folder, covariates_path, params_path, counterfactuals_folder, factuals_folder

N_DATASETS = 2592
N_DATASETS_PER_SIZE = 432
N_STR_TO_INT = {
    '1k': 1000,
    '2.5k': 2500,
    '5k': 5000,
    '10k': 10000,
    '25k': 25000,
    '50k': 50000,
    '1000': 1000,
    '2500': 2500,
    '5000': 5000,
    '10000': 10000,
    '25000': 25000,
    '50000': 50000,
}
VALID_N = N_STR_TO_INT.keys()
VALID_LINKS = {'linear', 'quadratic', 'cubic', 'poly', 'log', 'exp'}


def load_lbidd(n=5000, observe_counterfactuals=False, return_ites=False,
               return_ate=False, return_params_df=False, link='quadratic',
               degree_y=None, degree_t=None, n_shared_parents='median', i=0,
               dataroot=None,
               print_paths=True):
    """
    Load the LBIDD dataset that is specified

    :param n: size of dataset (1k, 2.5k, 5k, 10k, 25k, or 50k)
    :param observe_counterfactuals: if True, return double-sized dataset with
        both y0 (first half) and y1 (second half) observed
    :param return_ites: if True, return ITEs
    :param return_ate: if True, return ATE
    :param return_params_df: if True, return the DataFrame of dataset parameters
        that match
    :param link: link function (linear, quadratic, cubic, poly, log, or exp)
    :param degree_y: degree of function for Y (e.g. 1, 2, 3, etc.)
    :param degree_t: degree of function for T (e.g. 1, 2, 3, etc.)
    :param n_shared_parents: number covariates that T and Y share as causal parents
    :param i: index of parametrization to choose among the ones that match
    :return: dictionary of results
    """

    folder, scaling_zip, scaling_folder, covariates_path, params_path, counterfactuals_folder, factuals_folder = \
        get_paths(dataroot=dataroot)
    if print_paths:
        print(scaling_folder)
        print(covariates_path)

    # Check if files exist
    if not (os.path.isfile(scaling_zip) and os.path.isfile(covariates_path)):
        raise FileNotFoundError(
            'You must first download scaling.tar.gz and x.csv from '
            'https://www.synapse.org/#!Synapse:syn11738963 and put them in the '
            'datasets/lbidd/ folder. This requires creating an account on '
            'Synapse and accepting some terms and conditions.'
        )

    # Process dataset size (n)
    if n is not None:
        if not isinstance(n, str):
            n = str(n)
        if n.lower() not in VALID_N:
            raise ValueError('Invalid n: {} ... Valid n: {}'.format(n, list(VALID_N)))
        n = N_STR_TO_INT[n]

    # Unzip 'scaling.tar.gz' if not already unzipped
    if not os.path.exists(scaling_folder):
        print('Unzipping {} ...'.format(SCALING_TAR_ZIP), end=' ')
        tar = tarfile.open(scaling_zip, "r:gz")
        tar.extractall(folder)
        tar.close()
        print('DONE')

    # Load and filter the params DataFrame
    params_df = pd.read_csv(params_path)
    if n is not None:
        params_df = params_df[params_df['size'] == n]   # Select dataset size
    if link is not None:
        if link not in VALID_LINKS:
            raise ValueError('Invalid link function type: {} ... Valid links: {}'
                             .format(link, VALID_LINKS))
        if link == 'linear':
            link = 'poly'
            degree_y = 1
            degree_t = 1
        elif link == 'quadratic':
            link = 'poly'
            degree_y = 2
            degree_t = 2
        elif link == 'cubic':
            link = 'poly'
            degree_y = 3
            degree_t = 3
        params_df = params_df[params_df['link_type'] == link]   # Select link function
    if degree_y is not None:
        params_df = params_df[params_df['deg(y)'] == degree_y]  # Select degree Y
    if degree_t is not None:
        params_df = params_df[params_df['deg(z)'] == degree_t]  # Select degree T

    # Filter by number of parents that T and Y share
    valid_n_shared_parents = params_df['n_conf(yz)'].unique().tolist()
    if n_shared_parents in valid_n_shared_parents:
        params_df = params_df[params_df['n_conf(yz)'] == n_shared_parents]
    elif isinstance(n_shared_parents, str) and n_shared_parents.lower() == 'max':
        max_shared_parents = params_df['n_conf(yz)'].max()
        params_df = params_df[params_df['n_conf(yz)'] == max_shared_parents]
    elif isinstance(n_shared_parents, str) and n_shared_parents.lower() == 'median':
        median_i = len(params_df) // 2
        median_shared_parents = params_df['n_conf(yz)'].sort_values().iloc[median_i]
        params_df = params_df[params_df['n_conf(yz)'] == median_shared_parents]
    elif n_shared_parents is None:
        pass
    else:
        raise ValueError('Invalid n_shared_parents ... must be either None, "max", "median", or in {}'
                         .format(valid_n_shared_parents))

    if params_df.empty:
        raise ValueError('No datasets have that combination of parameters.')

    output = {}
    if return_params_df:
        output['params_df'] = params_df

    # Get ith dataset that has the right parameters
    if i < len(params_df):
        ufid = params_df['ufid'].iloc[i]
    else:
        raise ValueError('Invalid i: {} ... with that parameter combination, i must be an int such that 0 <= i < {}'
                         .format(i, len(params_df)))

    covariates_df = pd.read_csv(covariates_path, index_col=INDEX_COL_NAME)
    factuals_path = os.path.join(factuals_folder, ufid + FILE_EXT)
    factuals_df = pd.read_csv(factuals_path, index_col=INDEX_COL_NAME)
    joint_factuals_df = covariates_df.join(factuals_df, how='inner')

    output['t'] = joint_factuals_df['z'].to_numpy()
    output['y'] = joint_factuals_df['y'].to_numpy()
    output['w'] = joint_factuals_df.drop(['z', 'y'], axis='columns').to_numpy()

    if observe_counterfactuals or return_ites or return_ate:
        counterfactuals_path = os.path.join(counterfactuals_folder, ufid + COUNTERFACTUAL_FILE_SUFFIX + FILE_EXT)
        counterfactuals_df = pd.read_csv(counterfactuals_path, index_col=INDEX_COL_NAME)
        joint_counterfactuals_df = covariates_df.join(counterfactuals_df, how='inner')

        # Add t column and stack y0 potential outcomes and y1 potential outcomes in same df
        if observe_counterfactuals:
            joint_y0_df = joint_counterfactuals_df.drop(['y1'], axis='columns').rename(columns={'y0': 'y'})
            joint_y0_df['t'] = 0
            joint_y1_df = joint_counterfactuals_df.drop(['y0'], axis='columns').rename(columns={'y1': 'y'})
            joint_y1_df['t'] = 1
            stacked_y_counterfactuals_df = pd.concat([joint_y0_df, joint_y1_df])

            output['obs_counterfactual_t'] = stacked_y_counterfactuals_df['t'].to_numpy()
            output['obs_counterfactual_y'] = stacked_y_counterfactuals_df['y'].to_numpy()
            output['obs_counterfactual_w'] = stacked_y_counterfactuals_df.drop(['t', 'y'], axis='columns').to_numpy()

        if return_ites:
            ites = joint_counterfactuals_df['y1'] - joint_counterfactuals_df['y0']
            output['ites'] = ites.to_numpy()

        if return_ate:
            ites = joint_counterfactuals_df['y1'] - joint_counterfactuals_df['y0']
            output['ate'] = ites.to_numpy().mean()

    return output


def lbidd_iter(n=None, observe_counterfactuals=False, return_ites=False,
               return_ate=False, return_params_df=False, dataroot=None):
    """
    Iterator for LBIDD datasets of a given size of just all of them

    :param n: size of datasets to iterate over (1k, 2.5k, 5k, 10k, 25k, or 50k)
        if None, iterate over all 2592 datasets
    :param observe_counterfactuals: if True, return double-sized dataset with
        both y0 (first half) and y1 (second half) observed
    :param return_ites: if True, return ITEs
    :param return_ate: if True, return ATE
    :param return_params_df: if True, return the DataFrame of dataset parameters
        that match
    :yield: dictionary of results
    """

    folder, scaling_zip, scaling_folder, covariates_path, params_path, counterfactuals_folder, factuals_folder = \
        get_paths(dataroot=dataroot)

    if n is None:
        n_datasets = N_DATASETS
    else:
        if not isinstance(n, str):
            n = str(n)
        if n.lower() not in VALID_N:
            raise ValueError('Invalid n: {} ... Valid n: {}'.format(n, list(VALID_N)))
        n = N_STR_TO_INT[n]
    params_df = pd.read_csv(params_path)

    if n is not None:
        params_df = params_df[params_df['size'] == n]
        n_datasets = N_DATASETS_PER_SIZE
    for i in range(n_datasets):
        yield load_lbidd(n=n, observe_counterfactuals=observe_counterfactuals,
                         return_ites=return_ites, return_ate=return_ate,
                         return_params_df=return_params_df, link=None,
                         degree_y=None, degree_t=None, n_shared_parents=None,
                         i=i)
