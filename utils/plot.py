import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def scatterplot(data,
                label_names,
                column_names=None,
                domain_names=None,
                save_path=None,
                ax=None,
                marker='o',
                legend=True,
                tight=True,
                palette=None):
    df = pd.DataFrame(data, columns=column_names)
    df[r'$Class$'] = label_names
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 5))

    if domain_names is None:
        sns.scatterplot(x=df.columns[0],
                        y=df.columns[1],
                        hue=df.columns[2],
                        data=df,
                        ax=ax,
                        edgecolor='k',
                        marker=marker,
                        palette=palette,
                        legend=legend)
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
                        edgecolor='k',
                        palette=palette,
                        legend=legend)
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
    # out = ax.pcolormesh(xx, yy, Z, **params)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def plot_gamma_with_marginals(gamma, marg_x, marg_y, cmap='jet', height=6):
    n_x = len(marg_x)
    n_y = len(marg_y)
    g = sns.JointGrid(marg_x, marg_y, height=height)

    g.ax_marg_x.plot(np.arange(0, n_x), marg_x)
    g.ax_marg_y.plot(marg_y, np.arange(0, n_y))

    g.ax_joint.imshow(gamma, cmap=cmap)
    g.ax_joint.set_yticks([])
    g.ax_joint.set_xticks([])

    return g


def discrete_gamma_marg(gamma, mu, nu, cmap='jet', height=6, cbar=True):
    supp_x, marg_x = mu
    supp_y, marg_y = nu

    n_x = len(marg_x)
    n_y = len(marg_y)
    xx, yy = np.mgrid[0:n_x, 0:n_y]
    zz = np.zeros([len(xx), len(yy)])
    for xxi in xx:
        for yyj in yy:
            zz[yyj, xxi] = gamma[xxi, yyj]
    g = sns.JointGrid(marg_x, marg_y, height=height)
    g.ax_marg_x.bar(np.arange(0, n_x), marg_x, width=1, edgecolor='k', color="#ff7070")
    g.ax_marg_y.barh(width=marg_y, y=np.arange(0, n_y), height=1, edgecolor='k', color="#7391ff")

    g.ax_joint.set_xticks(np.arange(0, n_x))
    g.ax_joint.set_yticks(np.arange(0, n_y))
    g.ax_joint.grid(linestyle=":")
    im = g.ax_joint.imshow(gamma, cmap=cmap)

    if cbar:
        fig = g.fig
        ax = g.ax_marg_y
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')
    plt.tight_layout()

    return g


def generate_hierarchical_graph_json(nodes, faults, adjacency_matrix):
    hierarchical_graph = []
    for i, (src_node, fault) in enumerate(zip(nodes, faults)):
        hierarchical_graph.append({
            'name': "Sample.{}.{}".format(fault, src_node),
            'Class': fault,
            'Domain': src_node[:3],
            'imports': []
        })
        for j, (tgt_node, fault) in enumerate(zip(nodes, faults)):
            if adjacency_matrix[i, j] > 0:
                hierarchical_graph[-1]['imports'].append("Sample.{}.{}".format(fault, tgt_node))
    
    return hierarchical_graph


def generate_links_json(nodes, faults, adjacency_matrix):
    links = []
    for i, (src_node, fault) in enumerate(zip(nodes, faults)):
        for j, (tgt_node, fault) in enumerate(zip(nodes, faults)):
            if adjacency_matrix[i, j] > 0:
                links.append({
                    '{},{}'.format(src_node, tgt_node): adjacency_matrix[i, j]
                })
    return links