import pandas as pd
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors


import matplotlib.pyplot as plt



metric = 'ate_abs_bias' # ate_rmse / ate_abs_bias / mean_pehe / ate_std_error

ylabel = {
'ate_rmse': 'ATE RMSE',
'mean_pehe': 'PEHE',
'ate_abs_bias': '|ATE Bias|',
'ate_std_error': 'ATE Std Error'
}[metric]



fn = 'rmse_sorted_estimators.csv'
f = pd.read_csv(fn)
# f[['meta-estimator', 'outcome_model', 'prop_score_model']]


ATEs = (('lalonde_psid', -13235.31193079472),
        ('lalonde_cps', -7156.20333393101),
         ('twins', -0.06923672))

perf_cols = ['ate_rmse', 'mean_pehe', 'ate_abs_bias','ate_std_error']
for dataset, ate in ATEs:
    f.loc[f['dataset']==dataset, perf_cols] /= abs(ate)



iden = f['meta-estimator'] + ':' + f['outcome_model'].fillna('-') + ':' + f['prop_score_model'].fillna('-')
f['identifier'] = iden
perf = f.groupby('identifier').mean()[metric].dropna()



# don't change the order... otherwise it will change e.g. ipw / standardization first

# meta
perf.index = perf.index.str.replace('stratified_standardization', 'SS')
perf.index = perf.index.str.replace('ipw_trimeps.01', 'IPWT')
perf.index = perf.index.str.replace('ipw', 'IPW')
perf.index = perf.index.str.replace('standardization', 'S')


# model
perf.index = perf.index.str.replace('Standardized_SVM', 'StdSVM')
perf.index = perf.index.str.replace('Standardized_LinearSVM', 'StdLinSVM')
perf.index = perf.index.str.replace('LinearSVM', 'LinSVM')
perf.index = perf.index.str.replace('LogisticRegression', 'LogReg')
perf.index = perf.index.str.replace('LinearRegression', 'LinReg')
perf.index = perf.index.str.replace('liblinear', 'liblin')
perf.index = perf.index.str.replace('_', '-')



perf.index = perf.index.str.replace(':-', '')

########

#
# model_tags = ['LogisticRegression', 'kNN', 'DecisionTree', 'GaussianNB', 'QDA']
# colors = ['brown', 'red', 'green', 'blue', 'cyan']
# custom_lines = [Line2D([0], [0], color=color, lw=4) for color in colors]

# model_tags = f['meta-estimator'].unique().tolist()
model_tags = ['S', 'SS', 'IPW', 'IPWT']
model_tags_ = ['S: Standardization', 'SS: Stratified Standardization', 'IPW', r'IPWT: ($\epsilon=0.01$)']
tableau_colors = list(mcolors.TABLEAU_COLORS.keys())
custom_lines = [Line2D([0], [0], color=color, lw=4) for color in tableau_colors[:len(model_tags)]]

model2color = {k:v for k, v in zip(model_tags, tableau_colors[:len(model_tags)])}




models = perf.index.to_list()
# models = models.split('\n')
# perf = list(map(float, perf.split('\n')))
perf = perf.to_list()
colors = list()
for model in models:
    colors.append(model2color[model.split(':')[0]])



models_sorted, perf_sorted, colors_sorted  = zip(*sorted(zip(models, perf,colors), key=lambda z: z[1]))

fig = plt.figure(figsize=(15, 3))
ax0 = fig.add_subplot(111)
ax0.bar(models_sorted, perf_sorted, color=colors_sorted)
if 'pehe' in metric:
    ax0.set_xticklabels(models_sorted, rotation=45, fontsize=8, ha='right')
else:
    ax0.set_xticklabels(models_sorted, rotation=45, fontsize=6, ha='right')
plt.ylabel(ylabel, fontsize=15)


ax0.spines['top'].set_visible(False)
d = .005 # how big to make the diagonal lines in axes coordinates
kwargs = dict(transform=ax0.transAxes, color='k', clip_on=False, lw=0.5)
# ax2.plot((-d,d),(-d,+d), **kwargs) # top-right diagonal
ax0.plot((-d,d),(1-d,1+d), **kwargs) # bottom-right diagonal
ax0.plot((1-d,1+d),(1-d,1+d), **kwargs) # bottom-left diagonal
D = 0.02
ax0.plot((-d,d),(1-d+D,1+d+D), **kwargs) # bottom-right diagonal
ax0.plot((1-d,1+d),(1-d+D,1+d+D), **kwargs) # bottom-left diagonal


plt.yscale('log')
plt.ylim(0, 10)

if 'pehe' in metric:
    ax0.legend(custom_lines[:2], model_tags_[:2])
else:
    ax0.legend(custom_lines, model_tags_)

plt.tight_layout()
plt.savefig(f'compare_estimators_{metric}.pdf')
