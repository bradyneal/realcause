import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
# import seaborn_qqplot as sqp


def compare_joints(x1, y1, x2, y2, xlabel1=None, ylabel1=None, xlabel2=None, ylabel2=None, save_fname=None):
    f, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=[12, 5])
    sns.kdeplot(x1, y1, ax=ax[0])
    ax[0].set(xlabel=xlabel1, ylabel=ylabel1)
    sns.kdeplot(x2, y2, ax=ax[1])
    ax[1].set(xlabel=xlabel2, ylabel=ylabel2)
    save_and_show(f, save_fname)
    return f


def compare_marginals(x1, x2, label1=None, label2=None, ax=None):
    sns.distplot(x1, ax=ax, label=label1)
    sns.distplot(x2, ax=ax, label=label2)


def compare_bivariate_marginals(x1, x2, y1, y2, xlabel=None, ylabel=None, label1=None, label2=None, save_fname=None):
    f, ax = plt.subplots(1, 2, figsize=[12, 4])

    compare_marginals(x1, x2, label1=label1, label2=label2, ax=ax[0])
    ax[0].legend()
    ax[0].set(xlabel=xlabel, ylabel='Density')

    compare_marginals(y1, y2, label1=label1, label2=label2, ax=ax[1])
    ax[1].legend()
    ax[1].set(xlabel=ylabel)

    save_and_show(f, save_fname)
    return f


def save_and_show(f, save_fname):
    if save_fname is not None:
        f.savefig(save_fname, bbox_inches='tight')
    plt.show()
