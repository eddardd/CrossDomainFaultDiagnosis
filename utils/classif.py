import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

from utils.preprocessing import feature_normalization
from utils.preprocessing import feature_scaling


def eval_clf(clf, Xtr, ytr, Xts, yts, weights=None):
    if weights is not None:
        clf.fit(Xtr, ytr, sample_weight=weights)
    else:
        clf.fit(Xtr, ytr)
    yp = clf.predict(Xts)
    return clf, accuracy_score(yts, yp), confusion_matrix(yts, yp)

def cval_performance(clf, X, y, d, domain, baseline=False, normalization="scaling"):
    Xs, ys = X[np.where(d == 0)[0]], y[np.where(d == 0)[0]]
    Xt, yt = X[np.where(d == domain)[0]], y[np.where(d == domain)[0]]
    index_crossval = np.load('./data/crossval_index.npy')
    accs = []
    for fold in range(5):
        ind_s = np.intersect1d(np.where(index_crossval[0, :] != fold)[0],
                               np.where(index_crossval[0, :] != -1)[0])
        ind_t = np.intersect1d(np.where(index_crossval[domain, :] != fold)[0],
                               np.where(index_crossval[domain, :] != -1)[0])
        Xtr, ytr = Xs[ind_s, :], ys[ind_s]
        if normalization == "normalization":
            Xtr = feature_normalization(Xtr)
        elif normalization == "scaling":
            Xtr = feature_scaling(Xtr)
        Xts, yts = Xt[ind_t, :], yt[ind_t]
        if normalization == "normalization":
            Xts = feature_normalization(Xts)
        elif normalization == "scaling":
            Xts = feature_scaling(Xts)

        if baseline:
            clf.fit(Xtr, ytr)
            yp = clf.predict(Xts)
        else:
            # yts is passed, but it is not used.
            clf.fit(Xtr, ytr, Xts, yts)
            yp = clf.predict(Xts)
        accs.append(accuracy_score(yp, yts))

    return np.array(accs)

def cval_on_source(clf, Xs, ys, n_folds=5):
    source_skf = StratifiedKFold(n_splits=n_folds,
                                    random_state=None,
                                    shuffle=False)
    gen_source = source_skf.split(Xs, ys)
    accs = []
    for (ind_tr, ind_ts) in gen_source:
        Xtr, ytr = Xs[ind_tr], ys[ind_tr]
        Xts, yts = Xs[ind_ts], ys[ind_ts]

        clf.fit(Xtr, ytr)
        yp = clf.predict(Xts)
        accs.append(accuracy_score(yp, yts))

    return np.array(accs)