import numpy as np
from scipy.spatial.distance import cdist


def uLSIF(Xs, Xt, kernel='linear', kernel_param=None, basis_vectors=None, reg=0.0):
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
    ns = Xs.shape[0]
    nt = Xt.shape[0]
    if basis_vectors is None:
        ind = np.arange(nt)
        n_basis = np.maximum(100, int(0.5 * nt))
        basis_vectors = Xt[np.random.choice(ind, size=n_basis), :]
    else:
        n_basis = len(basis_vectors)
    Phi_s = phi(Xs, basis_vectors, kernel=kernel, kernel_param=kernel_param)
    Phi_t = phi(Xt, basis_vectors, kernel=kernel, kernel_param=kernel_param)
    H = np.einsum('ki,kj->ij', Phi_s, Phi_s) / ns
    h = np.einsum('ki->i', Phi_t) / nt
    
    alpha = np.dot(np.linalg.inv(H + reg * np.eye(n_basis)), h)
    return np.dot(Phi_s, alpha)

