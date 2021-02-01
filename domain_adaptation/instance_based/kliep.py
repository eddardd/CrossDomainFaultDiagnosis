import numpy as np
from scipy.spatial.distance import cdist


def KLIEP(Xs, Xt,
          kernel='linear',
          sigma=1.,
          numItermax=1000,
          epsilon=1e-4,
          percentSamples=0.1,
          support_vectors=None):
    """Implementation of Kullback-Leibler Importance Estimation Procedure (KLIEP) [1] algorithm

    Parameters
    ----------
    Xs : :class:`numpy.ndarray`
        Numpy array containing the source samples. Shape: (ns, nf), where ns is the number of source samples and
        nf is the number of features.
    Xt : :class:`numpy.ndarray`
        Numpy array containing the target samples. Shape: (nt, nf), where ns is the number of target samples and
        nf is the number of features.
    kernel : str
        Choice of kernel function. For linear, phi(x, y) = <x, y>, for rbf, phi(x, y) = exp(-||x-y||_2^2 / 2 (sigma^2))
    sigma : float
        Kernel parameter for rbf.
    numItermax : int
        Maximum number of iterations for the optimizer
    epsilon : float
        Optimization step
    percentSamples : float
        Percentage of samples used as support vectors in the algorithm's formulation
    support_vectors :
        Pre-set support vectors.

    References
    ----------
    [1] Sugiyama, Masashi, et al. "Direct importance estimation for covariate shift adaptation." Annals of the Institute
        of Statistical Mathematics 60.4 (2008): 699-746.
    """
    def cost(alpha, A, kernel='linear'):
        return np.mean(np.log(A @ alpha))

    def phi(X, support_vectors, sigma, kernel='linear'):
        if kernel == 'linear':
            return np.dot(X, support_vectors.T)
        elif kernel == 'rbf':
            return np.exp(- cdist(X, support_vectors) / (2 * (sigma ** 2)))
        else:
            raise ValueError('Bad kernel')

    
    nt = Xt.shape[0]
    
    if support_vectors is None:
        randind = np.random.choice(np.arange(nt), size=int(Xt.shape[0] * percentSamples))
        support_vectors = Xt[randind, :]
    
    A = phi(Xt, support_vectors, sigma=sigma, kernel=kernel)
    b = np.mean(phi(Xs, support_vectors, sigma=sigma, kernel=kernel), axis=0)

    alpha = np.ones(b.shape) / b.shape[0]
    costs = []
    
    k = 0
    while k < numItermax:
        k += 1
        alpha = alpha + epsilon * A.T @ (1 /  (A @ alpha))
        alpha = alpha + (1 - np.dot(b, alpha)) * b / np.dot(b, b)
        alpha = np.maximum(0, alpha)
        alpha = alpha / (np.dot(b, alpha))

        c = cost(alpha, A, kernel=kernel)
        if len(costs) > 0.0 and c < np.max(costs):
            break
        costs.append(c)
    weights = np.dot(alpha, phi(Xs, support_vectors, sigma=sigma, kernel=kernel))
    return weights