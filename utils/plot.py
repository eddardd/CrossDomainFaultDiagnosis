import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def scatterplot(data,
                label_names,
                column_names=None,
                domain_names=None,
                save_path=None,
                ax=None,
                marker='o',
                legend=True,
                tight=True):
    df = pd.DataFrame(data, columns=column_names)
    df[r'$Class$'] = label_names
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    if domain_names is None:
        sns.scatterplot(x=df.columns[0],
                        y=df.columns[1],
                        hue=df.columns[2],
                        data=df,
                        ax=ax,
                        edgecolor='k',
                        marker=marker)
        if legend: ax.legend(bbox_to_anchor=(1, 1))
        if tight: plt.tight_layout()        
    else:
        df[r'$Domain$'] = domain_names
        sns.scatterplot(x=df.columns[0],
                        y=df.columns[1],
                        hue=df.columns[2],
                        style=df.columns[3],
                        data=df,
                        ax=ax,
                        edgecolor='k')
        if legend: ax.legend(bbox_to_anchor=(1, 1))
        if tight: plt.tight_layout()


    return ax


def make_grid(x, y, h=.02):
    x_min, x_max = x.min() - .25, x.max() + .25
    y_min, y_max = y.min() - .25, y.max() + .25
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.pcolormesh(xx, yy, Z, **params)
    return out