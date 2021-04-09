import numpy as np


class FOPTD:
    def __init__(self):
        self.alpha = None
        self.fitted = False

    def __phi(self, i, t, y, dt):
        vec = np.zeros([1, 3])
        vec[0, 0] = sum(y[:i]) * dt
        vec[0, 1] = t[i]
        vec[0, 2] = -1

        return vec

    def fit(self, t, y):
        dt = t[1] - t[0]
        Phi = np.concatenate([self.__phi(i, t, y, dt) for i in range(len(t))], axis=0)
        self.alpha = np.linalg.pinv(Phi) @ (y.reshape(-1, 1))
        self.fitted = True

    def predict(self):
        if self.fitted:
            tau = - self.alpha[0, 0] ** (-1)
            K = self.alpha[1, 0] * tau
            theta = self.alpha[2, 0] / K

            return [tau, K, theta]
        else:
            raise ValueError('Trying to use predict in a unfitted model.')