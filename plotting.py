import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import warnings


FIGSIZE = [12, 5]


def compare_joints(x1, y1, x2, y2, xlabel1=None, ylabel1=None, xlabel2=None, ylabel2=None,
                   name=None, save_fname=None, kwargs=None):
    f, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=FIGSIZE)
    if name is not None:
        f.suptitle(name + ' Joint Kernel Density Estimate Plots')

    sns.kdeplot(x1, y1, ax=ax[0], **kwargs)
    ax[0].set(xlabel=xlabel1, ylabel=ylabel1)
    sns.kdeplot(x2, y2, ax=ax[1], **kwargs)
    ax[1].set(xlabel=xlabel2, ylabel=ylabel2)
    save_and_show(f, save_fname)
    return f


def compare_marginal_hists(x1, x2, label1=None, label2=None, ax=None):
    sns.distplot(x1, ax=ax, label=label1)
    sns.distplot(x2, ax=ax, label=label2)


def compare_marginal_qqplots(x1, x2, label1=None, label2=None, ax=None):
    try:
        from statsmodels.graphics.gofplots import qqplot_2samples
    except ImportError as e:
        warnings.warn('statsmodels is not installed, so no qqplot will be made '
                     '\nInstall: pip install statsmodels', Warning)
        return
    if len(x1) > len(x2):
        print('Unexpected behavior: switching the order of the arguments to qqplot to avoid statsmodels error...',
              '\n\t"{}" will be on the x-axis instead of y-axis and "{}" will be on the y-axis instead of x-axis'
              .format(label2, label1))
        return qqplot_2samples(x2, x1, xlabel=label2, ylabel=label1, line='45', ax=ax)
    else:
        return qqplot_2samples(x1, x2, xlabel=label1, ylabel=label2, line='45', ax=ax)


def compare_bivariate_marginals(x1, x2, y1, y2, xlabel=None, ylabel=None, label1=None, label2=None,
                                hist=True, qqplot=True, name=None, save_hist_fname=None, save_qq_fname=None):
    if not (hist or qqplot):
        print('Both hist and qqplot are False, so no plots were made.')

    plots = []
    if hist:
        f1, ax1 = plt.subplots(1, 2, figsize=FIGSIZE)
        if name is not None:
            f1.suptitle(name + ' Marginal Histograms')

        compare_marginal_hists(x1, x2, label1=label1, label2=label2, ax=ax1[0])
        ax1[0].legend()
        ax1[0].set(xlabel=xlabel, ylabel='Density')

        compare_marginal_hists(y1, y2, label1=label1, label2=label2, ax=ax1[1])
        ax1[1].legend()
        ax1[1].set(xlabel=ylabel)

        save_and_show(f1, save_hist_fname)
        plots.append(f1)

    if qqplot:
        f2, ax2 = plt.subplots(1, 2, figsize=FIGSIZE)
        if name is not None:
            f2.suptitle(name + ' Marginal Q-Q Plots')

        compare_marginal_qqplots(x1, x2, ax=ax2[0],
                                 label1=get_quantile_label(label1, xlabel),
                                 label2=get_quantile_label(label2,  xlabel))

        compare_marginal_qqplots(y1, y2, ax=ax2[1],
                                 label1=get_quantile_label(label1, ylabel),
                                 label2=get_quantile_label(label2, ylabel))
        save_and_show(f2, save_qq_fname)
        plots.append(f2)

    if len(plots) == 1:
        return plots[0]
    else:
        return plots


def get_quantile_label(dist, var):
    return '{} {} quantiles'.format(dist, var)


def save_and_show(f, save_fname):
    if save_fname is not None:
        f.savefig(save_fname, bbox_inches='tight')
    plt.show()
