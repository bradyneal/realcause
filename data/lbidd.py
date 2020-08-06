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

folder = os.path.join(DATA_FOLDER, 'lbidd')
scaling_zip = os.path.join(folder, 'scaling.tar.gz')
scaling_folder = os.path.join(folder, 'scaling')
covariates_path = os.path.join(folder, 'x.csv')
params_path = os.path.join(scaling_folder, 'params.csv')
counterfactuals_folder = os.path.join(scaling_folder, 'counterfactuals')
factuals_folder = os.path.join(scaling_folder, 'factuals')

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


def load_lbidd(n=5000, observe_counterfactuals=False, return_ites=False, i=0):
    """
    Load the LBIDD dataset that is specified

    :param n: size of dataset (1k, 2.5k, 5k, 10k, 25k, or 50k)
    :param observe_counterfactuals: if True, return double-sized dataset with
        both y0 (first half) and y1 (second half) observed
    :param return_ites: if True, return ITEs
    :param i: which parametrization to choose (0 <= i < 432)
    :return: dictionary of results
    """
    # Check if files exist
    if not (os.path.isfile(scaling_zip) and os.path.isfile(covariates_path)):
        raise FileNotFoundError(
            'You must first download scaling.tar.gz and x.csv from '
            'https://www.synapse.org/#!Synapse:syn11738963 and put them in the '
            'datasets/lbidd/ folder. This requires creating an account on '
            'Synapse and accepting some terms and conditions.'
        )

    # Process dataset size (n)
    if not isinstance(n, str):
        n = str(n)
    n = n.lower()
    if n.lower() not in VALID_N:
        raise ValueError('Invalid n: {} ... Valid n: {}'.format(n, list(VALID_N)))
    n = N_STR_TO_INT[n]

    # Make sure arguments are valid
    if not (isinstance(i, int) and 0 <= i < N_DATASETS_PER_SIZE):
        raise ValueError('Invalid i: {} ... Valid i: 0 <= i < {}'.format(i, N_DATASETS_PER_SIZE))

    # Unzip 'scaling.tar.gz' if not already unzipped
    if not os.path.exists(scaling_folder):
        print('Unzipping {} ...'.format(SCALING_TAR_ZIP), end=' ')
        tar = tarfile.open(scaling_zip, "r:gz")
        tar.extractall(folder)
        tar.close()
        print('DONE')

    params_df = pd.read_csv(params_path)

    # Take ith dataset that has the right size
    ufid = params_df[params_df['size'] == n]['ufid'].iloc[i]

    covariates_df = pd.read_csv(covariates_path, index_col=INDEX_COL_NAME)
    factuals_path = os.path.join(factuals_folder, ufid + FILE_EXT)
    factuals_df = pd.read_csv(factuals_path, index_col=INDEX_COL_NAME)
    joint_factuals_df = covariates_df.join(factuals_df, how='inner')

    output = {}
    output['t'] = joint_factuals_df['z'].to_numpy()
    output['y'] = joint_factuals_df['y'].to_numpy()
    output['w'] = joint_factuals_df.drop(['z', 'y'], axis='columns').to_numpy()

    if observe_counterfactuals or return_ites:
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

    return output
