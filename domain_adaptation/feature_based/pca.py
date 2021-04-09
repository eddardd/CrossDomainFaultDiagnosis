import numpy as np
from sklearn.decomposition import PCA


def DAPCA(Xs, Xt, n_components=2):
    return PCA(n_components=n_components).fit(np.concatenate([Xs, Xt], axis=0)).components_.T