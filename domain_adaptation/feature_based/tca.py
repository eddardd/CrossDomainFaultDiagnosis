import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from scipy.sparse.linalg import eigs


def TCA(Xs, Xt, kernel="linear", n_components=2, mu=1.0, gamma=1.0):
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
    elif kernel == 'tanh':
        Kss = np.tanh(gamma * np.dot(Xs, Xs.T))
        Kst = np.tanh(gamma * np.dot(Xs, Xt.T))
        Ktt = np.tanh(gamma * np.dot(Xt, Xt.T))
    else:
        raise ValueError('Bad kernel')
        
    K = np.vstack([np.hstack([Kss, Kst]),
                   np.hstack([Kst.T, Ktt])])
    
    L = np.vstack([
        np.hstack([np.ones([ns, ns]) / (ns ** 2), - np.ones([ns, nt]) / (ns * nt)]),
        np.hstack([- np.ones([nt, ns]) / (ns * nt), np.ones([nt, nt]) / (nt * nt)])
    ])

    H = np.eye(ns + nt) - np.ones([ns + nt, ns + nt]) / (ns + nt)

    J = np.dot(np.linalg.pinv(np.eye(ns + nt) + mu * np.dot(np.dot(K, L), K)), np.dot(np.dot(K, H), K))
    _, C = eigs(J, k=n_components)

    return np.real(C), K
