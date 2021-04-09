import ot
import numpy as np


class MSE:
    def __call__(self, Yp, Ys):
        L = ot.dist(Yp, Ys, metric='sqeuclidean')

        return L
    
    def __repr__(self):
        return 'MeanSquaredError'

    def __format__(self, format_spec):
        return format('MeanSquaredError', format_spec)

    
class CategoricalCrossEntropy:
    def __call__(self, Yp, Ys):
        ns, k = Ys.shape
        nt, _ = Yp.shape
        L = np.zeros([nt, ns])
        log_pred = np.log(Yp)
        L = np.einsum('jk,ik->ij', Ys, log_pred) / k

        return L

    def __repr__(self):
        return 'CategoricalCrossEntropy'

    def __format__(self, format_spec):
        return format('CategoricalCrossEntropy', format_spec)

    
class HingeLoss:
    def __call__(self, Y, F):
        res = np.zeros((Y.shape[0], F.shape[0]))
        for i in range(Y.shape[1]):
            res += np.maximum(0, 1 - Y[:,i].reshape((Y.shape[0],1)) * F[:,i].reshape((1, F.shape[0]))) ** 2
        return res

    def __repr__(self):
        return 'HingeLoss'

    def __format__(self, format_spec):
        return format('HingeLoss', format_spec)