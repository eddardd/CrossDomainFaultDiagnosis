from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def eval_clf(clf, Xtr, ytr, Xts, yts, weights=None):
    if weights is not None:
        clf.fit(Xtr, ytr, sample_weight=weights)
    else:
        clf.fit(Xtr, ytr)
    yp = clf.predict(Xts)
    return clf, accuracy_score(yts, yp), confusion_matrix(yts, yp)