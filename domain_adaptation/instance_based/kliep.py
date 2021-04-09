import numpy as np
from scipy.spatial.distance import cdist


def KLIEP(Xs, Xt,
          kernel='linear',
          kernel_param=None,
          numItermax=1000,
          epsilon=1e-4,
          percentSamples=0.1,
          basis_vectors=None):
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

    def phi(X, basis_vectors,  kernel='linear', kernel_param=None):
        if kernel == 'linear':
            return np.dot(X, basis_vectors.T)
        elif kernel == 'rbf':
            D = cdist(X, basis_vectors)
            if kernel_param is None:
                kernel_param = [1 / np.median(D)]
            return np.exp(- kernel_param[0] * D)
        elif kernel == 'tanh':
            if kernel_param is None:
                kernel_param = [1.0, 0.0]
            return np.tanh(kernel_param[0] * np.dot(X, basis_vectors.T) + kernel_param[1])
        else:
            raise ValueError('Bad kernel')

    
    nt = Xt.shape[0]
    if basis_vectors is None:
        ind = np.arange(nt)
        n_basis = np.maximum(100, int(0.5 * nt))
        basis_vectors = Xt[np.random.choice(ind, size=n_basis), :]
    else:
        n_basis = len(basis_vectors)
    
    A = phi(Xt, basis_vectors, kernel_param=kernel_param, kernel=kernel)
    b = np.mean(phi(Xs, basis_vectors, kernel_param=kernel_param, kernel=kernel), axis=0)

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
    weights = np.dot(phi(Xs, basis_vectors, kernel_param=kernel_param, kernel=kernel), alpha)
    return weights