import matplotlib
matplotlib.use('Agg')
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import warnings
import os
import numpy as np
from PIL import Image


FIGSIZE = [12, 5]
SINGLE_FIGSIZE = [5.5, 5]
DIR = '../plots'
DPI = 300


def compare_joints(x1, y1, x2, y2, xlabel1=None, ylabel1=None, xlabel2=None, ylabel2=None,
                   xlabel=None, ylabel=None, label1=None, label2=None, title=True,
                   name='', save_fname=None, test=False, kwargs=None):
    uniq1 = np.unique(x1)
    uniq2 = np.unique(x2)
    n_uniq1 = len(uniq1)
    n_uniq2 = len(np.unique(x2))

    if n_uniq1 == 2 and n_uniq2 == 2:
        f, ax = plt.subplots(1, 2, sharex=False, sharey=True, figsize=FIGSIZE)
        if not np.array_equal(uniq1, [0, 1]):
            raise ValueError('Binary x1 that is not [0, 1]: {}'.format(uniq1))
        if not np.array_equal(uniq2, [0, 1]):
            raise ValueError('Binary x2 that is not [0, 1]: {}'.format(uniq2))

        compare_marginal_hists(y1[x1 == 0], y2[x2 == 0], label1=label1, label2=label2, ax=ax[0])
        ax[0].legend()
        ax[0].set(xlabel=ylabel, ylabel='p({} | {} = 0)'.format(ylabel, xlabel.upper()))

        compare_marginal_hists(y1[x1 == 1], y2[x2 == 1], label1=label1, label2=label2, ax=ax[1])
        ax[1].legend()
        ax[1].set(xlabel=ylabel, ylabel='p({} | {} = 1)'.format(ylabel, xlabel.upper()))

    elif n_uniq1 == len(x1) and n_uniq2 == len(x2):
        f, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=FIGSIZE)
        sns.kdeplot(x1, y1, ax=ax[0], **kwargs)
        ax[0].set(xlabel=xlabel1, ylabel=ylabel1)
        sns.kdeplot(x2, y2, ax=ax[1], **kwargs)
        ax[1].set(xlabel=xlabel2, ylabel=ylabel2)
    else:
        raise ValueError('x1 and x2 have unexpected number of unique elements: {} and {}'
                         .format(n_uniq1, n_uniq2))
    if title:
        f.suptitle(name + ' Joint Kernel Density Estimate Plots')
    save_and_show(f, save_fname, test=test)

    return f


def compare_marginal_hists(x1, x2, label1=None, label2=None, ax=None):
    if is_binary(x1, x2):
        sns.distplot(x1, kde=False, ax=ax, label=label1)
        sns.distplot(x2, kde=False, ax=ax, label=label2)
    else:
        try:
            sns.distplot(x1, ax=ax, label=label1)
        except RuntimeError:
            sns.distplot(x1, ax=ax, label=label1, kde_kws={'bw': 0.5})
        try:
            sns.distplot(x2, ax=ax, label=label2)
        except RuntimeError:
            sns.distplot(x1, ax=ax, label=label1, kde_kws={'bw': 0.5})


def is_binary(x1, x2=None):
    x1_is_binary = len(np.unique(x1)) == 2
    return x1_is_binary if x2 is None else \
        x1_is_binary and len(np.unique(x2)) == 2


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
                                hist=True, qqplot=True, title=True, name='',
                                save_hist_fname=None, save_qq_fname=None,
                                test=False):
    if not (hist or qqplot):
        print('Both hist and qqplot are False, so no plots were made.')

    plots = []
    if hist:
        f1, ax1 = plt.subplots(1, 2, figsize=FIGSIZE)
        if title:
            f1.suptitle(name + ' Marginal Histograms')

        compare_marginal_hists(x1, x2, label1=label1, label2=label2, ax=ax1[0])
        ax1[0].legend()
        ax1[0].set(xlabel=xlabel, ylabel='Density')

        compare_marginal_hists(y1, y2, label1=label1, label2=label2, ax=ax1[1])
        ax1[1].legend()
        ax1[1].set(xlabel=ylabel)

        save_and_show(f1, save_hist_fname, test=test)
        plots.append(f1)

    if qqplot is 'both':
        f2, ax2 = plt.subplots(1, 2, figsize=FIGSIZE)
        if title:
            f2.suptitle(name + ' Marginal Q-Q Plots')

        compare_marginal_qqplots(x1, x2, ax=ax2[0],
                                 label1=get_quantile_label(label1, xlabel),
                                 label2=get_quantile_label(label2,  xlabel))

        compare_marginal_qqplots(y1, y2, ax=ax2[1],
                                 label1=get_quantile_label(label1, ylabel),
                                 label2=get_quantile_label(label2, ylabel))

        save_and_show(f2, save_qq_fname, test=test)
        plots.append(f2)
    elif qqplot is 'y' or qqplot:
        f2, ax2 = plt.subplots(1, 1, figsize=SINGLE_FIGSIZE)
        compare_marginal_qqplots(y1, y2, ax=ax2,
                                 label1=get_quantile_label(label1, ylabel),
                                 label2=get_quantile_label(label2, ylabel))
        if title:
            plt.title(name + ' Y Marginal Q-Q Plot')
        save_and_show(f2, save_qq_fname, test=test)
        plots.append(f2)

    if len(plots) == 1:
        return plots[0]
    else:
        return plots


def get_quantile_label(dist, var):
    return '{} {} quantiles'.format(dist, var)


def save_and_show(f, save_fname, dir=DIR, test=False):
    if test:
        return
    if save_fname is not None:
        if not os.path.exists(dir):
            os.makedirs(dir)
        if os.sep not in save_fname:
            save_fname = os.path.join(dir, save_fname)
        f.savefig(save_fname, bbox_inches='tight', dpi=DPI)
    
    plt.show()
    plt.close()


# Source: http://193.51.245.4/tutorials/convert_a_matplotlib_figure
def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()
 
    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


def fig2img(fig):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data(fig)
    w, h, d = buf.shape
    return Image.frombytes("RGBA", (w, h), buf.tostring())
