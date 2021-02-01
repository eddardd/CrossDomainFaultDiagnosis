import numpy as np
from cvxopt import matrix, solvers
from sklearn.metrics.pairwise import rbf_kernel


def KMM(Xs, Xt, kernel='linear', B=1.0, gamma=None, epsilon=None, verbose=False):
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
    if epsilon is None:
        epsilon = B / np.sqrt(ns)
    if kernel == 'linear':
        Kss = np.dot(Xs, Xs.T)
        Kst = np.dot(Xs, Xt.T)
    elif kernel == 'rbf':
        Kss = rbf_kernel(Xs, Xs, gamma)
        Kst = rbf_kernel(Xs, Xt, gamma)
    else:
        raise ValueError('Bad kernel')
        
    P = matrix(Kss / (ns ** 2))
    q = matrix((nt / ns) * np.sum(Kst, axis=1) * (2 / (nt ** 2)))
    G = matrix(np.vstack([np.ones([1, ns]),
                          np.ones([1, ns]),
                          np.diag([1] * ns),
                          np.diag([-1] * ns)]))
    h = matrix(np.array([ns * (1 + epsilon)] + [ns * (epsilon - 1)] + [1] * ns + [0] * ns))
    
    sol = solvers.qp(P=P, q=-q, G=G, h=h)
    return np.squeeze(np.array(sol['x']))