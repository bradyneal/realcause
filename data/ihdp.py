"""
File for loading the IHDP semi-synthetic dataset.

Hill (2011) took the real covariates from the IHDP data and
generated semi-synthetic data by generating the outcomes via random functions
("response surfaces"). Response surface A corresponds to a linear function; we
do not provide that data in this file. Response surface B corresponds to a
nonlinear function; this is what we provide in this file. We get it from Shalit
et al. (2017) who get it from the NPCI (Dorie, 2016) R package.

References:

    Dorie, V. (2016). NPCI: Non-parametrics for Causal Inference.
        https://github.com/vdorie/npci

    Hill, J. (2011). Bayesian Nonparametric Modeling for Causal Inference.
        Journal of Computational and Graphical Statistics, 20:1, 217-240.

    Shalit, U., Johansson, F.D. & Sontag, D. (2017). Estimating individual
        treatment effect: generalization bounds and algorithms. Proceedings of
        the 34th International Conference on Machine Learning.
"""

import numpy as np
from utils import download_dataset, unzip

IHDP_100_TRAIN_URL = 'http://www.fredjo.com/files/ihdp_npci_1-100.train.npz'
IHDP_100_TEST_URL = 'http://www.fredjo.com/files/ihdp_npci_1-100.test.npz'
IHDP_1000_TRAIN_URL = 'http://www.fredjo.com/files/ihdp_npci_1-1000.train.npz.zip'
IHDP_1000_TEST_URL = 'http://www.fredjo.com/files/ihdp_npci_1-1000.test.npz.zip'

SPLIT_OPTIONS = {'train', 'test', 'all'}
N_REALIZATIONS_OPTIONS = {100, 1000}


def load_ihdp(split='all', i=0, observe_counterfactuals=False, return_ites=False,
              return_ate=False, dataroot=None):
    """
    Load a single instance of the IHDP dataset

    :param split: 'train', 'test', or 'both' (the default IHDP split is 90/10)
    :param i: dataset instance (0 <= i < 1000)
    :param observe_counterfactuals: if True, return double-sized dataset with
        both y0 (first half) and y1 (second half) observed
    :param return_ites: if True, return ITEs
    :param return_ate: if True, return ATE
    :return: dictionary of results
    """
    if 0 <= i < 100:
        n_realizations = 100
    elif 100 <= i < 1000:
        n_realizations = 1000
        i = i - 100
    else:
        raise ValueError('Invalid i: {} ... Valid i: 0 <= i < 1000'.format(i))

    if split == 'all':
        train, test = load_ihdp_datasets(split=split, n_realizations=n_realizations,
                                         dataroot=dataroot)
    else:
        data = load_ihdp_datasets(split=split, n_realizations=n_realizations,
                                  dataroot=dataroot)

    ws = []
    ts = []
    ys = []
    ys_cf = []
    itess = []
    datasets = [train.f, test.f] if split == 'all' else [data.f]
    for dataset in datasets:
        w = dataset.x[:, :, i]
        t = dataset.t[:, i]
        y = dataset.yf[:, i]
        y_cf = dataset.ycf[:, i]
        ites = dataset.mu1[:, i] - dataset.mu0[:, i]

        ws.append(w)
        ts.append(t)
        ys.append(y)
        ys_cf.append(y_cf)
        itess.append(ites)

    w = np.vstack(ws)
    t = np.concatenate(ts)
    y = np.concatenate(ys)
    y_cf = np.concatenate(ys_cf)
    ites = np.concatenate(itess)
    ate = np.mean(ites)

    d = {}
    if observe_counterfactuals:
        d['w'] = np.vstack([w, w.copy()])
        d['t'] = np.concatenate([t, np.logical_not(t.copy()).astype(int)])
        d['y'] = np.concatenate([y, y_cf])
        ites = np.concatenate([ites, ites.copy()])  # comment if you don't want duplicates
    else:
        d['w'] = w
        d['t'] = t
        d['y'] = y

    if return_ites:
        d['ites'] = ites
    if return_ate:
        d['ate'] = ate

    return d


def load_ihdp_datasets(split='train', n_realizations=100, dataroot=None):
    """
    Load the IHDP data with the nonlinear response surface ("B") that was used
    by Shalit et al. (2017). Description of variables:
        x: covariates (25: 6 continuous and 19 binary)
        t: treatment (binary)
        yf: "factual" (observed) outcome
        ycf: "counterfactual" outcome (random)
        mu0: noiseless potential outcome under control
        mu1: noiseless potential outcome under treatment
        ate: I guess just what was reported in the Hill (2011) paper...
            Not actually accurate. The actual SATEs for the data are the
            following (using (mu1 - mu0).mean()):
                train100:   4.54328871735309
                test100:    4.269906127209613
                all100:     4.406597422281352

                train1000:  4.402550421661204
                test1000:   4.374712690625632
                all1000:    4.388631556143418
        yadd: ???
        ymul: ???

    :param split: 'train', 'test', or 'both'
    :param n_realizations: 100 or 1000 (the two options that the data source provides)
    :return: NpzFile with all the data ndarrays in the 'f' attribute
    """
    if split.lower() not in SPLIT_OPTIONS:
        raise ValueError('Invalid "split" option {} ... valid options: {}'
                         .format(split, SPLIT_OPTIONS))
    if isinstance(n_realizations, str):
        n_realizations = int(n_realizations)
    if n_realizations not in N_REALIZATIONS_OPTIONS:
        raise ValueError('Invalid "n_realizations" option {} ... valid options: {}'
                         .format(n_realizations, N_REALIZATIONS_OPTIONS))
    if n_realizations == 100:
        if split == 'train' or split == 'all':
            path = download_dataset(IHDP_100_TRAIN_URL, 'IHDP train 100', dataroot=dataroot)
            train = np.load(path)
        if split == 'test' or split == 'all':
            path = download_dataset(IHDP_100_TEST_URL, 'IHDP test 100', dataroot=dataroot)
            test = np.load(path)
    elif n_realizations == 1000:
        if split == 'train' or split == 'all':
            path = download_dataset(IHDP_1000_TRAIN_URL, 'IHDP train 1000', dataroot=dataroot)
            unzip_path = unzip(path)
            train = np.load(unzip_path)
        if split == 'test' or split == 'all':
            path = download_dataset(IHDP_1000_TEST_URL, 'IHDP test 1000', dataroot=dataroot)
            unzip_path = unzip(path)
            test = np.load(unzip_path)

    if split == 'train':
        return train
    elif split == 'test':
        return test
    elif split == 'all':
        return train, test
