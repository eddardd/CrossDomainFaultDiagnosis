import uuid
import numpy as np
import tensorflow as tf
from keras import backend as K
from sklearn.metrics import accuracy_score


@tf.custom_gradient
def GradientReversal(x):
	def grad(dy):
		return -1 * dy
	return x, grad

class GradientReversalLayer(tf.keras.layers.Layer):
	def __init__(self):
		super(GradientReversalLayer, self).__init__()
		
	def call(self, inputs):
		return GradientReversal(inputs)


class EvalOnDomains(tf.keras.callbacks.Callback):
    def __init__(self, model, Xtargets, ytargets):
        self.model = model
        self.Xtargets = Xtargets
        self.ytargets = ytargets
        self.ndomains = len(self.Xtargets)
    
    def on_epoch_end(self, epoch, logs=None):
        if logs is None: logs = {}
        for domain in range(self.ndomains):
            Xts, yts = self.Xtargets[domain], self.ytargets[domain]
            Yp = self.model.predict(Xts)
            yp = Yp.argmax(axis=1) + np.min(yts)
            acc = accuracy_score(yts, yp)
            logs['dom{}_acc'.format(domain)] = acc
            print('Domain: {}, Acc: {}'.format(domain, acc))
