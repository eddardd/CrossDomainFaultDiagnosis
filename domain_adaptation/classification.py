import ot
import numpy as np

# Instance-based algorithms
from domain_adaptation.instance_based.kmm import KMM
from domain_adaptation.instance_based.kliep import KLIEP

# OT-Based algorithms
from domain_adaptation.ot_based.jdot import fit_jdot, hinge_loss
from domain_adaptation.ot_based.ot_laplace import SinkhornLaplaceTransport


class OTClassifier:
    def __init__(self, clf, ot_solver=ot.da.EMDTransport, **kwargs):
        self.clf = clf
        self.ot_solver = ot_solver(**kwargs)

    def fit(self, Xs, ys, Xt, yt=None):
        self.tXs = self.ot_solver.fit_transform(Xs, ys, Xt, yt)

        self.clf.fit(self.tXs, ys)
        return self

    def predict(self, X):
        return self.clf.predict(X)


class JDOTClassifier:
    def __init__(self, clf, loss=hinge_loss, alpha=.1, reg=1, method='emd', metric='sqeuclidean', numItermax=10):
        self.alpha = alpha
        self.clf = clf
        self.loss = loss
        self.alpha = alpha
        self.reg = reg
        self.method = method
        self.metric = metric
        self.numItermax = numItermax
        self.logs = None

    def fit(self, Xs, ys, Xt, yt=None):
        nt = Xt.shape[0]
        G, Ys, logs = fit_jdot(Xs=Xs,
                               Xt=Xt,
                               ys=ys,
                               yt=yt,
                               clf=self.clf,
                               loss=self.loss,
                               alpha=self.alpha,
                               reg=self.reg,
                               method=self.method,
                               metric=self.metric,
                               numItermax=self.numItermax)
        self.G = G
        self.logs = logs
        Ps = nt * (self.G.T).dot(Ys)
        # Note: use np.min(ys) in case classes in ys don't start with 0
        self.clf.fit(Xt, (Ps.argmax(axis=1) + np.min(ys)))

        return self

    def predict(self, X):
        return self.clf.predict(X)


class WeightedClassifier:
    def __init__(self, clf, importance_estimator, **kwargs):
        self.clf = clf
        self.importance_estimator = importance_estimator
        self.kwargs = kwargs

    def fit(self, Xs, ys, Xt, yt=None):
        self.weights = self.importance_estimator(Xs, Xt, **self.kwargs)
        self.clf.fit(Xs, ys, sample_weight=self.weights)

        return self

    def predict(self, X):
        return self.clf.predict(X)
