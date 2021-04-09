import numpy as np
from pygsvd import gsvd
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from scipy.linalg import null_space


def GFK(Xs, Xt, ys=None, n_components=2, projection='pca'):
    if projection == 'pca':
        Bs = PCA(n_components=n_components).fit(Xs).components_.T
        Bt = PCA(n_components=n_components).fit(Xt).components_.T
    elif projection == 'pls' and ys is not None:
        Bs = PLSRegression(n_components=n_components).fit(Xs, ys).x_loadings_
        Bt = PCA(n_components=n_components).fit(Xt).components_.T
    else:
        if projection == 'pls':
            raise ValueError('For PLS projection, expected ys to be not None.')
        else:
            raise ValueError("Expected 'projection' to be either 'pca' or 'pls', but got {}".format(projection))
    Rs = null_space(Bs.T)

    A = np.dot(Bs.T, Bt)
    B = np.dot(Rs.T, Bt)
    [Gamma, Sigma, V, U1, U2] = gsvd(A, B)

    # Sanity Check
    # assert np.sum(abs(A - np.dot(U1, np.dot(np.diag(Gamma), V.T)))) < 1e-10
    # assert np.sum(abs(B - np.dot(U2, np.dot(np.diag(Sigma), V.T)))) < 1e-10
    
    principal_angles = np.arccos(np.diag(Gamma))
    
    L1 = 1 + np.sin(2 * principal_angles) / (2 * principal_angles)
    L2 = (np.cos(2 * principal_angles) - 1) / (2 * principal_angles)
    L3 = 1 - np.sin(2 * principal_angles) / (2 * principal_angles)

    L = np.vstack([
        np.hstack([L1, L2]),
        np.hstack([L2, L3]),
    ])

    M = np.hstack([np.dot(Bs, U1), np.dot(Rs, U2)])
    G = np.dot(M, np.dot(L, M.T))

    return G