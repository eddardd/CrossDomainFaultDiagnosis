import numpy as np
from cvxopt import matrix, solvers
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import rbf_kernel


def KMM(Xs, Xt, kernel='rbf', kernel_param=None, B=1000, symmetrize=True, verbose=False):
    """ Implementation of Kernel Mean Matching algorithm [1] using CVXOPT.

    Parameters
    ----------
    Xs : :class:`numpy.ndarray`
        Numpy array containing the source samples. Shape: (ns, nf), where ns is the number of source samples and
        nf is the number of features.
    Xt : :class:`numpy.ndarray`
        Numpy array containing the target samples. Shape: (nt, nf), where ns is the number of target samples and
        nf is the number of features.
    kernel : str
        Either linear of rbf. In the first case, k(x, y) = <x, y>. In the second case, k(x, y) = exp(- gamma ||x - y||²)
    B : float
        KMM Hyperparameter. The weights will belong to the interval [0, B],
    epsilon : float
        KMM hyperparameter. The weights mean will belong to [epsilon - 1, epsilon + 1]

    References
    ----------
    [1] Gretton, Arthur, et al. "Covariate shift by kernel mean matching." Dataset shift in machine learning 3.4
        (2009): 5.
    """
    solvers.options['show_progress'] = verbose
    ns = Xs.shape[0]
    nt = Xt.shape[0]
    if kernel == 'rbf':
        if kernel_param is None:
            D = cdist(Xs, Xs)
            gamma = np.median(D)
        else:
            gamma = kernel_param[0]
        Kss = rbf_kernel(Xs, Xs, gamma)
        Kss = 0.5 * (Kss + Kss.T)
        Kst = rbf_kernel(Xs, Xt, gamma)
    elif kernel == 'linear':
        Kss = np.dot(Xs, Xs.T)
        Kst = np.dot(Xs, Xt.T)
    elif kernel == 'tanh':
        if kernel_param is None:
            a = np.median(cdist(Xs, Xs)) ** -1
            b = 0.0
        else:
            a = kernel_param[0]
            b = kernel_param[1]
        Kss = np.tanh(a * np.dot(Xs, Xs.T) + b)
        Kst = np.tanh(a * np.dot(Xs, Xt.T) + b)
    ones_nt = np.ones(shape=(nt, 1))
    kappa = np.dot(Kst, ones_nt)
    kappa = -(ns / nt) * kappa
    eps = (np.sqrt(ns) - 1) / np.sqrt(ns)

    # constraints
    A0 = np.ones(shape=(1, ns))
    A1 = -np.ones(shape=(1, ns))
    A = np.vstack([A0, A1, -np.eye(ns), np.eye(ns)])
    b = np.array([[ns * (eps + 1), ns * (eps - 1)]])
    b = np.vstack([b.T, -np.zeros(shape=(ns, 1)), np.ones(shape=(ns, 1)) * B])

    P = matrix(Kss.astype(np.double), tc='d')
    q = matrix(kappa.astype(np.double), tc='d')
    G = matrix(A.astype(np.double), tc='d')
    h = matrix(b.astype(np.double), tc='d')
    return np.squeeze(np.array(solvers.qp(P, q, G, h)['x']))