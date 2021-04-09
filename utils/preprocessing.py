import numpy as np


def feature_scaling(X, a=0.0, b=1.0):
    Xmax = np.max(X, axis=0)
    Xmin = np.min(X, axis=0)

    return (b - a) * ((X - Xmin) / (Xmax - Xmin)) - a


def feature_normalization(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    return (X - mean) / std