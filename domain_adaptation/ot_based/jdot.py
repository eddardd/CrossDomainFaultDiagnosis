import ot
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


def hinge_loss(Y, F):
    res = np.zeros((Y.shape[0],F.shape[0]))
    for i in range(Y.shape[1]):
        res += np.maximum(0, 1-Y[:,i].reshape((Y.shape[0],1))*F[:,i].reshape((1,F.shape[0])))**2
    return res


def fit_jdot(Xs, Xt, ys, yt=None,
             clf=LinearSVC,
             loss=hinge_loss,
             alpha=.1,
             reg=1,
             method='emd',
             metric='sqeuclidean',
             numItermax=10):
    ns = Xs.shape[0]
    nt = Xt.shape[0]

    # Build label matrix
    classes = np.unique(ys)
    Ys = np.zeros([ns, len(classes)])
    for i, c in enumerate(classes):
        Ys[:, i] = (ys == c)

    a = ot.unif(ns)
    b = ot.unif(nt)

    C0 = ot.dist(Xs, Xt, metric=metric)
    C0 = C0 / np.max(C0)
    C = alpha * C0.copy()
    log = {'C': [], 'G': [], 'tL': [], 'cL': [], 'Acc': []}

    k = 0
    while k < numItermax:
        k += 1
        # Step 1. solve OT problem
        if method == 'sinkhorn':
            G = ot.sinkhorn(a, b, C, reg)
        elif method == 'emd':
            G = ot.emd(a, b, C)
        else:
            raise ValueError('Unsupported method: {}'.format(method))
        # Step 2. Build proportion matrix
        Ps = nt * (G.T).dot(Ys)
        
        # Step 3. Fit classifier
        clf.fit(Xt, (Ps.argmax(axis=1) + 1))
        
        # Step 4. Calculate loss
        Yp = clf.decision_function(Xt)
        yp = Yp.argmax(axis=1) + 1
        L = loss(Yp, Ps)
        
        C = alpha * C0.copy() + L

        log['C'].append(C)
        log['G'].append(G)
        log['tL'].append(np.sum(C0 * G))
        log['cL'].append(np.sum(C * G))
        if yt is not None:
            log['Acc'].append(accuracy_score(yt, yp))

    return G, Ys, log