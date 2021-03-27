import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, spearmanr, pearsonr
import numpy as np

RESULTS_DIR = Path('results')

stand_df = pd.concat([
    pd.read_csv(RESULTS_DIR / 'psid_cps_twins_standard.csv'),
    pd.read_csv(RESULTS_DIR / 'psid_cps_twins_strat_standard.csv'),
], axis=0).reset_index(drop=True)
ipw_df = pd.concat([
    pd.read_csv(RESULTS_DIR / 'psid_cps_twins_ipw.csv'),
    pd.read_csv(RESULTS_DIR / 'psid_cps_twins_ipw_trim_01.csv'),
], axis=0).reset_index(drop=True)
full_df = pd.concat([stand_df, ipw_df], axis=0).reset_index(drop=True)
complete_df = pd.concat([stand_df, ipw_df, pd.read_csv(RESULTS_DIR / 'psid_cps_twins_ipw_stabilized.csv')], axis=0).reset_index(drop=True)
complete_df.to_csv('causal-predictive-analysis.csv', index=False)


def get_correlation(df, causal_score, regression_score='mean_test_neg_root_mean_squared_error',
                    classification_score='mean_test_f1', corr_func=spearmanr, only_corr=True):
    # get sklearn predictive scores and negate them so that lower is better
    stand_idx = df['meta-estimator'].str.contains('standardization')
    ipw_idx = df['meta-estimator'].str.startswith('ipw')
    pred_stand = pd.Series(dtype='float64')
    if stand_idx.any():
        pred_stand = -df[stand_idx][regression_score]
    pred_ipw = pd.Series(dtype='float64')
    if ipw_idx.any():
        pred_ipw = pred_stand.append(-df[ipw_idx][classification_score])

    # append the predictive scores together
    pred = pred_stand.append(pred_ipw)
    assert len(pred) == len(df)

    assert not pred.isna().any()

    # get causal scores and remove NaNs
    causal = df[causal_score]
    na_causal = causal.isna()
    if na_causal.any():
        print('Number {} NaNs: {}'.format(causal_score, na_causal.sum()))
        not_na_idx = ~na_causal
        causal = causal[not_na_idx]
        pred = pred[not_na_idx]

    corr_obj = corr_func(pred, causal)
    if corr_func is pearsonr:
        return corr_obj[0]
    else:
        try:
            return corr_obj.correlation if only_corr else corr_obj
        except AttributeError:
            return corr_obj


def prob_same_sign(x, y, allow_eq_zero=True, allow_zeros=False):
    assert len(x) == len(y)
    n_total = len(x) * (len(x) - 1) / 2
    n_check = 0
    n_same_sign = 0
    for i in range(len(x)):
        for j in range(i + 1, len(y)):
            # x_sign = np.sign(x[i] - x[j])
            # y_sign = np.sign(y[i] - y[j])
            x_sign = np.sign(x.iloc[i] - x.iloc[j])
            y_sign = np.sign(y.iloc[i] - y.iloc[j])
            if x_sign == 0 == y_sign:
                if allow_eq_zero:
                    n_same_sign += 1
            elif x_sign == 0 or y_sign == 0:
                if allow_zeros:
                    n_same_sign += 1
            elif x_sign == y_sign:
                n_same_sign += 1

            n_check += 1
    assert n_total == n_check
    return n_same_sign / n_total


def prob_better_better(x, y):
    return prob_same_sign(x, y, allow_eq_zero=False, allow_zeros=False)


def prob_better_or_equal(x, y):
    return prob_same_sign(x, y, allow_eq_zero=True, allow_zeros=True)


CORRELATION_MEAURES = [
    ('spearman', spearmanr),
    ('kendall', kendalltau),
    ('pearson', pearsonr),
    ('prob_better_better', prob_better_better),
    ('prob_same_sign', prob_same_sign),
    ('prob_better_or_equal', prob_better_or_equal)
]



def to_list(x):
    if isinstance(x, list):
        return x
    else:
        return [x]


def get_correlation_df(df, grouping, correlation_measures=CORRELATION_MEAURES,
                       causal_score=['ate_rmse'], regression_score=['mean_test_neg_root_mean_squared_error'],
                       classification_score=['mean_test_f1']):
    causal_scores = to_list(causal_score)
    regression_scores = to_list(regression_score)
    classification_scores = to_list(classification_score)
    d = {**{col: [] for col in grouping},
         **{name: [] for name, _ in correlation_measures},
         'causal_score': [], 'reg_score': [], 'class_score': []}
    for group_name, group in df.groupby(grouping):
        if len(group) <= 1:
            continue
        for causal_score in causal_scores:
            for regression_score in regression_scores:
                for classification_score in classification_scores:
                    d['causal_score'].append(causal_score)
                    d['reg_score'].append(regression_score)
                    d['class_score'].append(classification_score)
                    if not isinstance(group_name, tuple):
                        group_name = (group_name,)
                    for i, val in enumerate(group_name):
                        d[grouping[i]].append(val)
                    for measure_name, measure in correlation_measures:
                        corr = get_correlation(group, causal_score=causal_score, regression_score=regression_score,
                                               classification_score=classification_score, corr_func=measure)
                        d[measure_name].append(corr)
                        # if np.isnan(corr):
                        #     d[measure_name].append('nan')
    return pd.DataFrame(d)


def get_cv_df(df, model_type, cv_metric):
    valid_model_types = ['outcome_model', 'prop_score_model']
    if model_type not in valid_model_types:
        raise ValueError('Invalid mode_type {} ... Vald model types: {}'.format(model_type, valid_model_types))
    best_idx = df.groupby(['dataset', 'meta-estimator', model_type])[cv_metric].idxmax(axis=0)
    cv_df = df.iloc[best_idx].sort_values(by=['dataset', 'meta-estimator', model_type])
    return cv_df


outcome_df = get_correlation_df(stand_df, grouping=['dataset', 'meta-estimator', 'outcome_model'], causal_score=['ate_rmse', 'mean_pehe'])
outcome_df.drop(['reg_score', 'class_score'], axis='columns').to_csv('results/outcome_model_correlations.csv', float_format='%.2f', index=False)
outcome_summary = outcome_df.groupby(['dataset', 'causal_score'])['spearman', 'prob_better_or_equal'].median()

prop_df = get_correlation_df(ipw_df, grouping=['dataset', 'meta-estimator', 'prop_score_model'], causal_score=['ate_rmse'], classification_score=['mean_test_f1', 'mean_test_average_precision', 'mean_test_balanced_accuracy'])
prop_df.drop(['causal_score', 'reg_score'], axis='columns').to_csv('results/prop_score_model_correlations.csv', float_format='%.2f', index=False)
prop_summary = prop_df.groupby(['dataset'])['spearman', 'prob_better_or_equal'].median()

stand_cv_df = get_cv_df(df=stand_df, model_type='outcome_model', cv_metric='mean_test_neg_root_mean_squared_error')
dataset_meta_stand_df = get_correlation_df(stand_cv_df, grouping=['dataset', 'meta-estimator'], causal_score=['ate_rmse', 'mean_pehe'])
dataset_stand_df = get_correlation_df(stand_cv_df, grouping=['dataset'], causal_score=['ate_rmse', 'mean_pehe'])
dataset_stand_df.drop(['reg_score', 'class_score', 'prob_same_sign'], axis='columns').to_csv('results/dataset_stand_correlations.csv', float_format='%.2f', index=False)


def get_ipw_dataset_df_from_cv_metric(cv_metric, ipw_df=ipw_df):
    ipw_cv_df = get_cv_df(df=ipw_df, model_type='prop_score_model', cv_metric=cv_metric)
    # best_ipw_idx = ipw_df.groupby(['dataset', 'meta-estimator', 'prop_score_model'])[cv_metric].idxmax(axis=0)
    # ipw_cv_df = ipw_df.iloc[best_ipw_idx].sort_values(by=['dataset', 'meta-estimator', 'prop_score_model'])
    dataset_meta_ipw_df = get_correlation_df(ipw_cv_df, grouping=['dataset', 'meta-estimator'],
                                             causal_score=['ate_rmse'],
                                             classification_score=['mean_test_f1', 'mean_test_average_precision',
                                                                   'mean_test_balanced_accuracy'])
    dataset_ipw_df = get_correlation_df(ipw_cv_df, grouping=['dataset'], causal_score=['ate_rmse'],
                                        classification_score=['mean_test_f1', 'mean_test_average_precision',
                                                              'mean_test_balanced_accuracy'])
    return dataset_ipw_df, dataset_meta_ipw_df


dataset_ipw_f1_df, dataset_meta_f1_ipw_df = get_ipw_dataset_df_from_cv_metric('mean_test_f1')
dataset_ipw_prec_df, dataset_meta_prec_ipw_df = get_ipw_dataset_df_from_cv_metric('mean_test_average_precision')
dataset_ipw_acc_df, dataset_meta_acc_ipw_df = get_ipw_dataset_df_from_cv_metric('mean_test_balanced_accuracy')

dataset_ipw_f1_df.drop(['reg_score', 'causal_score', 'prob_same_sign'], axis='columns').to_csv('results/dataset_ipw_f1_correlations.csv', float_format='%.2f', index=False)
dataset_ipw_prec_df.drop(['reg_score', 'causal_score', 'prob_same_sign'], axis='columns').to_csv('results/dataset_ipw_prec_correlations.csv', float_format='%.2f', index=False)
dataset_ipw_acc_df.drop(['reg_score', 'causal_score', 'prob_same_sign'], axis='columns').to_csv('results/dataset_ipw_acc_correlations.csv', float_format='%.2f', index=False)


ipw_cv_df = get_cv_df(df=ipw_df, model_type='prop_score_model', cv_metric='mean_test_average_precision')

cv_df = pd.concat([stand_cv_df, ipw_cv_df])
rmse_cv_df = cv_df.sort_values(by=['dataset', 'ate_rmse', 'mean_pehe', 'ate_abs_bias'])[['dataset', 'meta-estimator', 'outcome_model', 'prop_score_model', 'ate_rmse', 'mean_pehe', 'ate_abs_bias', 'ate_std_error']]
bias_cv_df = cv_df.sort_values(by=['dataset', 'ate_abs_bias', 'ate_rmse', 'mean_pehe'])[['dataset', 'meta-estimator', 'outcome_model', 'prop_score_model', 'ate_abs_bias', 'ate_rmse', 'mean_pehe', 'ate_std_error']]
pehe_cv_df = stand_cv_df.sort_values(by=['dataset', 'mean_pehe', 'ate_rmse', 'ate_abs_bias'])[['dataset', 'meta-estimator', 'outcome_model', 'mean_pehe', 'ate_rmse', 'ate_abs_bias', 'ate_std_error']]

rmse_cv_df.to_csv('results/rmse_sorted_estimators.csv', float_format='%.2f', index=False)
bias_cv_df.to_csv('results/bias_sorted_estimators.csv', float_format='%.2f', index=False)
pehe_cv_df.to_csv('results/pehe_sorted_estimators.csv', float_format='%.2f', index=False)
