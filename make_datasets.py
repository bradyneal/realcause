import pandas as pd
import numpy as np
from numpy.testing import assert_approx_equal
from pathlib import Path
import time

from loading import load_from_folder

from data.lalonde import load_lalonde
from data.twins import load_twins
from consts import REALCAUSE_DATASETS_FOLDER, N_SAMPLE_SEEDS, N_AGG_SEEDS

FOLDER = Path(REALCAUSE_DATASETS_FOLDER)
FOLDER.mkdir(parents=True, exist_ok=True)

psid_gen_model, args = load_from_folder(dataset='lalonde_psid1')
cps_gen_model, args = load_from_folder(dataset='lalonde_cps1')
twins_gen_model, args = load_from_folder(dataset='twins')

psid_w, psid_t, psid_y = load_lalonde(obs_version='psid', data_format='pandas')
cps_w, cps_t, cps_y = load_lalonde(obs_version='cps', data_format='pandas')
d_twins = load_twins(data_format='pandas')
twins_w, twins_t, twins_y = d_twins['w'], d_twins['t'], d_twins['y']

gen_models = [psid_gen_model, cps_gen_model, twins_gen_model]
w_dfs = [psid_w, cps_w, twins_w]
names = ['lalonde_psid', 'lalonde_cps', 'twins']

dfs = []
print('N samples:', N_SAMPLE_SEEDS)
print('N seeds per sample:', N_AGG_SEEDS)
for gen_model, w_df, name in zip(gen_models, w_dfs, names):
    w_orig = w_df.to_numpy()

    dataset_start = time.time()
    print('Dataset:', name)
    ate_means = []
    for sample_i in range(N_SAMPLE_SEEDS):
        print('Sample:', sample_i)
        start_seed = sample_i * N_AGG_SEEDS
        end_seed = start_seed + N_AGG_SEEDS
        ates = []
        for seed in range(start_seed, end_seed):
            # print('Seed:', seed)
            w, t, (y0, y1) = gen_model.sample(w_orig, ret_counterfactuals=True, seed=seed)
            y = y0 * (1 - t) + y1 * t

            # w_errors = np.abs(w_orig - w)
            # assert w_errors.max() < 1e-2
            df = w_df
            df['t'] = t
            df['y'] = y
            df['y0'] = y0
            df['y1'] = y1
            df['ite'] = y1 - y0

            ate = (y1 - y0).mean()
            ates.append(ate)

        ate_means.append(np.mean(ates))
        df.to_csv(FOLDER / (name + '_sample{}.csv'.format(sample_i)), index=False)
        dfs.append(df)

    print('ATE mean mean (min-max) (std): {} ({} - {}) ({})'
          .format(np.mean(ate_means), np.min(ate_means), np.max(ate_means), np.std(ate_means)))
    print('Time elapsed: {} minutes'.format((time.time() - dataset_start) / 60))


# assert_approx_equal(psid_t.mean(), dfs[0]['t'].mean(), significant=1)
# assert_approx_equal(psid_y.mean(), dfs[0]['y'].mean(), significant=2)
# assert_approx_equal(cps_t.mean(), dfs[1]['t'].mean(), significant=1)
# assert_approx_equal(cps_y.mean(), dfs[1]['y'].mean(), significant=3)
# assert_approx_equal(twins_t.mean(), dfs[2]['t'].mean(), significant=2)
# assert_approx_equal(twins_y.mean(), dfs[2]['y'].mean(), significant=2)
