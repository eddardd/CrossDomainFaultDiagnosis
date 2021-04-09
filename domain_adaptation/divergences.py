import ot
import numpy as np

from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split

from pygsvd import gsvd
from scipy.linalg import null_space


def wasserstein_distance(Xs, Xt, solver="emd", metric='sqeuclidean', norm='max', **kwargs):
    """Calculates the Wasserstein distance between source and target data points

    """
    a = ot.unif(Xs.shape[0])
    b = ot.unif(Xt.shape[0])
    C = ot.dist(Xs, Xt, metric=metric)
    C = ot.utils.cost_normalization(C, norm=norm)
    if solver == "emd":
        G = ot.emd(a, b, C, **kwargs)
    elif solver == "sinkhorn":
        G = ot.sinkhorn(a, b, C, **kwargs)
    else:
        raise ValueError("Expected 'solver' to be either 'emd' or 'sinkhorn', but got {}".format(solver))
    
    return np.sum(C * G)


def h_divergence(Xs, Xt, clf, nruns=20):
    X = np.concatenate([Xs, Xt], axis=0)
    d = np.array([0] * len(Xs) + [1] * len(Xt))
    mae = 0
    for _ in range(nruns):
        Xtr, Xts, ytr, yts = train_test_split(X, d, train_size=0.8, stratify=d)
        clf.fit(Xtr, ytr)
        yp = clf.predict(Xts)
        mae += mean_absolute_error(yts, yp) / nruns
    return 2 * (1 - mae)


def maximum_mean_discrepancy(Xs, Xt, kernel='linear', gamma=None):
    ns = Xs.shape[0]
    nt = Xt.shape[0]
    if kernel == 'linear':
        Kss = np.dot(Xs, Xs.T)
        Kst = np.dot(Xs, Xt.T)
        Ktt = np.dot(Xt, Xt.T)
    elif kernel == 'rbf':
        Kss = rbf_kernel(Xs, Xs, gamma)
        Kst = rbf_kernel(Xs, Xt, gamma)
        Ktt = rbf_kernel(Xt, Xt, gamma)
    else:
        raise ValueError('Bad kernel')
    K = np.vstack([
        np.hstack([Kss, Kst]),
        np.hstack([Kst.T, Ktt])
    ])
        
    L = np.vstack([
        np.hstack([np.ones([ns, ns]) / (ns ** 2), - np.ones([ns, nt]) / (ns * nt)]),
        np.hstack([- np.ones([nt, ns]) / (ns * nt), np.ones([nt, nt]) / (nt * nt)])
    ])
    
    return np.trace(K * L)


def subspace_disagreement_measure(Xs, Xt, return_position=True):
    def calculate_principal_angles(A, B):
        _, S, _ = np.linalg.svd(np.dot(A.T, B))
        return np.arccos(S)
    
    n_components = np.min([Xs.shape[0], Xt.shape[0], Xs.shape[1], Xt.shape[1]])
    Bs = PCA(n_components=n_components).fit(Xs).components_.T
    Bt = PCA(n_components=n_components).fit(Xt).components_.T
    BsBt = PCA(n_components=n_components).fit(np.concatenate([Xs, Xt])).components_.T
    alpha_d = calculate_principal_angles(Bs, BsBt)
    beta_d = calculate_principal_angles(Bt, BsBt)

    sdm = .5 * (np.sin(alpha_d) + np.sin(beta_d))

    if return_position:
        return np.min(sdm), np.argmin(sdm)
    return np.min(sdm)


def ranking_of_domain(Xs, Xt, n_components=None):
    if n_components is None:
        n_components = Xs.shape[1] - 1
    Bs = PCA(n_components=n_components).fit(Xs).components_.T
    Rs = null_space(Bs.T)
    Bt = PCA(n_components=n_components).fit(Xt).components_.T
    A = np.dot(Bs.T, Bt)
    B = np.dot(Rs.T, Bt)
    Gamma, _, V, U1, _ = gsvd(A, B)
    s = np.dot(Bs, U1)
    t = np.dot(Bt, V)
    Gamma[np.where(np.isclose(Gamma, 1) == True)[0]] = 1
    angles = np.arccos(Gamma)

    var_s = np.diag(np.dot(np.dot(s.T, Xs.T), np.dot(s.T, Xs.T).T) / Xs.shape[0]) ** 2
    var_t = np.diag(np.dot(np.dot(t.T, Xt.T), np.dot(t.T, Xt.T).T) / Xt.shape[0]) ** 2

    return np.mean(angles * (.5 * (var_s / var_t) + .5 * (var_t / var_s) - 1))


def compute_mass_flow(G, ys, yt):
    n_classes = len(np.unique(ys))
    mass_flow = np.zeros([n_classes, n_classes])
    
    for i in range(n_classes):
        for j in range(n_classes):
            mass_flow[i, j] = np.sum(G[np.where(ys == i)[0], :][:, np.where(yt == j)[0]])
            
    return mass_flow


def undesired_mass_flow_index(G, ys, yt):
    Phi = compute_mass_flow(G, ys, yt)
    return np.sum(Phi - np.diag(np.diag(Phi)))