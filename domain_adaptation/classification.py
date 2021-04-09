import ot
import numpy as np
import tensorflow as tf

from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import rbf_kernel

# Feature-based algorithm
from domain_adaptation.feature_based.tca import TCA
from domain_adaptation.feature_based.gfk import GFK
from domain_adaptation.feature_based.pca import DAPCA

# OT-Based algorithms
from domain_adaptation.ot_based.jdot import fit_jdot
from domain_adaptation.metrics import HingeLoss
from domain_adaptation.ot_based.ot_laplace import SinkhornLaplaceTransport

# utils for DANN and CNNs
from domain_adaptation.utils import GradientReversal


class OTClassifier:
    def __init__(self, clf, ot_solver=ot.da.EMDTransport, **kwargs):
        self.clf = clf
        self.ot_solver = ot_solver(**kwargs)

    def fit(self, Xs, ys, Xt, yt=None, **kwargs):
        try:
            self.tXs = self.ot_solver.fit_transform(Xs, ys, Xt, None)
        except AttributeError:
            self.tXs = self.ot_solver.fit(Xs, ys, Xt, None).transform(Xs)

        self.clf.fit(self.tXs, ys, **kwargs)
        return self

    def predict(self, X):
        return self.clf.predict(X)


class JDOTClassifier:
    def __init__(self, clf, loss=HingeLoss(), alpha=.1, reg=1, method='emd', metric='sqeuclidean', numItermax=10):
        self.alpha = alpha
        self.clf = clf
        self.loss = loss
        self.alpha = alpha
        self.reg = reg
        self.method = method
        self.metric = metric
        self.numItermax = numItermax
        self.logs = None

    def fit(self, Xs, ys, Xt, yt=None, **kwargs):
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
        self.clf.fit(Xt, (Ps.argmax(axis=1) + np.min(ys)), **kwargs)

        return self

    def predict(self, X):
        return self.clf.predict(X)


class WeightedClassifier:
    def __init__(self, clf, importance_estimator, **kwargs):
        self.clf = clf
        self.importance_estimator = importance_estimator
        self.kwargs = kwargs

    def fit(self, Xs, ys, Xt, yt=None, **kwargs):
        self.weights = self.importance_estimator(Xs, Xt, **self.kwargs)
        self.clf.fit(Xs, ys, sample_weight=self.weights, **kwargs)

        return self

    def predict(self, X):
        return self.clf.predict(X)


class PCAClassifier:
    def __init__(self, clf, n_components=2):
        self.clf = clf
        self.n_components = n_components

    def fit(self, Xs, ys, Xt, yt=None, **kwargs):
        self.projectionM = DAPCA(Xs, Xt, n_components=self.n_components)
        Zs = np.dot(Xs, self.projectionM)

        self.clf.fit(Zs, ys, **kwargs)

        return self

    def predict(self, X):
        Z = np.dot(X, self.projectionM)

        return self.clf.predict(Z)


class TCAClassifier:
    def __init__(self, clf, kernel="linear", n_components=2, mu=1.0, gamma=1.0, normalize_projections='scale'):
        self.clf = clf
        self.kernel = kernel
        self.n_components = n_components
        self.mu = mu
        self.gamma = gamma
        self.normalize_projections = normalize_projections

    def fit(self, Xs, ys, Xt, yt=None, **kwargs):
        self.Xs = Xs.copy()
        self.C, self.K = TCA(Xs, Xt,
                             kernel=self.kernel,
                             n_components=self.n_components,
                             mu=self.mu,
                             gamma=self.gamma)
        Zs = np.dot(self.K[:Xs.shape[0], :], self.C)
        
        if self.normalize_projections == 'scale':
            Zs = (Zs - np.min(Zs, axis=0)) / (np.max(Zs, axis=0) - np.min(Zs, axis=0))
        elif self.normalize_projections == 'std':
            Zs = (Zs - np.mean(Zs, axis=0)) / (np.std(Zs, axis=0) + 1e-11)

        self.clf.fit(Zs, ys, **kwargs)

        return self

    def predict(self, X):
        ns = self.Xs.shape[0]
        Z = np.dot(self.K[ns:, :], self.C)

        return self.clf.predict(Z)


class GFKClassifier:
    def __init__(self, clf, n_components=2, projection='pca'):
        self.clf = clf
        self.n_components = n_components
        self.projection = projection

    def fit(self, Xs, ys, Xt, yt=None, **kwargs):
        G = GFK(Xs, Xt, ys=ys,
                n_components=self.n_components,
                projection=self.projection)
        self.projectionM = np.dot(G, Xs.T)
        Zs = np.dot(Xs, self.projectionM)
        self.clf.fit(Zs, ys, **kwargs)

        return self

    def predict(self, X):
        Z = np.dot(X, self.projectionM)

        return self.clf.predict(Z)



class CNNClassifier:
    def __init__(self, architecture_fn=None, loss=None, lr=1e-3, optimizer=None, batch_size=128, **kwargs):
        self.batch_size = batch_size
        self.loss = loss
        self.optimizer = optimizer
        self.lr = lr
        if architecture_fn is None:
            self.model = self.__build_model(**kwargs)
        else:
            self.model = architecture_fn(**kwargs)
        
        self.fitted = False

    def __build_model(self, input_shape=(1500, 1), n_classes=5):
        inp = tf.keras.layers.Input(shape=input_shape)
        cl1 = tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1,
                                    padding='same', activation='relu')(inp)
        pl1 = tf.keras.layers.MaxPool1D(pool_size=2)(cl1)

        cl2 = tf.keras.layers.Conv1D(filters=64, kernel_size=6, strides=1,
                                    padding='same', activation='relu')(pl1)
        pl2 = tf.keras.layers.MaxPool1D(pool_size=2)(cl2)

        cl3 = tf.keras.layers.Conv1D(filters=128, kernel_size=6, strides=1,
                                    padding='same', activation='relu')(pl2)
        pl3 = tf.keras.layers.MaxPool1D(pool_size=2)(cl3)

        representation = tf.keras.layers.Flatten()(pl3)

        fc1 = tf.keras.layers.Dense(units=1200, activation='relu')(representation)
        fc2 = tf.keras.layers.Dense(units=200, activation='relu')(fc1)
        fc3 = tf.keras.layers.Dense(units=n_classes, activation='softmax')(fc2)

        model = tf.keras.models.Model(inputs=inp, outputs=fc3)
        self.representation_model = tf.keras.models.Model(inputs=inp, outputs=fc1)
        return model

    def __compile_model(self, loss, lr, optimizer):
        if loss is None:
            loss = tf.keras.losses.CategoricalCrossentropy()
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(lr=lr)
        
        self.model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    def fit(self, Xs, ys, **kwargs):
        self.__compile_model(self.loss, self.lr, self.optimizer)
        Ys = tf.keras.utils.to_categorical(ys) if ys.ndim == 1 else ys
        self.history = self.model.fit(x=Xs, y=Ys, **kwargs)
        self.fitted = True

        """
        inp = self.model.layers[0].input
        outp = self.model.layers[-1].output
        max_outp = tf.keras.backend.max(outp, axis=1)
        saliency = tf.keras.backend.gradients(tf.keras.backend.sum(max_outp), inp)[0]
        max_class = tf.keras.backend.argmax(outp, axis=1)
        self.saliency_fn = tf.keras.backend.function([inp], [saliency, max_class])
        """

        return self.history
        
    def predict(self, X):
        Yp = self.model.predict(X)

        return Yp.argmax(axis=1)

    def get_representation(self, X):
        Z = self.representation_model.predict(X)

        return Z


class DomainAdversarialNN:
    def __init__(self, architecture_fn=None, loss=None, lr=1e-3, optimizer=None, batch_size=128,
                 input_shape=(1500, 1), n_classes=5):
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.__build_model()
        self.__compile_model(loss, lr, optimizer)
    
    def __build_model(self):
        # Feature Branch
        inp = tf.keras.layers.Input(shape=self.input_shape)
        cl1 = tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1,
                                    padding='same', activation='relu')(inp)
        pl1 = tf.keras.layers.MaxPool1D(pool_size=2)(cl1)

        cl2 = tf.keras.layers.Conv1D(filters=64, kernel_size=6, strides=1,
                                    padding='same', activation='relu')(pl1)
        pl2 = tf.keras.layers.MaxPool1D(pool_size=2)(cl2)

        cl3 = tf.keras.layers.Conv1D(filters=128, kernel_size=6, strides=1,
                                    padding='same', activation='relu')(pl2)
        pl3 = tf.keras.layers.MaxPool1D(pool_size=2)(cl3)

        representation = tf.keras.layers.Flatten()(pl3)
        
        # Classification branch
        branch_label = tf.keras.layers.Lambda(
            lambda x: tf.keras.backend.switch(tf.keras.backend.learning_phase(),
                                              tf.keras.backend.concatenate([x[:int(self.batch_size//2)],
                                                                            x[:int(self.batch_size//2)]], axis=0),
                                              x), output_shape=lambda x: x[0:])(representation)
        fc1 = tf.keras.layers.Dense(units=1200, activation='relu')(branch_label)
        fc2 = tf.keras.layers.Dense(units=200, activation='relu')(fc1)
        fc3 = tf.keras.layers.Dense(units=self.n_classes, activation='softmax', name='classifier_output')(fc2)
        
        # Domain prediction
        fc4 = tf.keras.layers.Dense(2, activation='softmax', name='domain_output')(fc3)

        # Building models
        # 1. Classifier model
        self.class_clf = tf.keras.models.Model(inputs=inp, outputs=[fc3])
        
        # 2. Domain Classifier model
        self.domain_clf = tf.keras.models.Model(inputs=inp, outputs=[fc4])

        # 3. Combined model
        self.dann_model = tf.keras.models.Model(inputs=inp, outputs=[fc4, fc3])

        # 4. Embeddings model
        self.embeddings_model = tf.keras.models.Model(inputs=inp, outputs=representation)

    def __compile_model(self, loss, lr, optimizer):
        if loss is None:
            loss = {'classifier_output': tf.keras.losses.CategoricalCrossentropy(),
                    'domain_output': tf.keras.losses.CategoricalCrossentropy()}
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(lr=lr)
               
        self.dann_model.compile(loss=loss, loss_weights={'classifier_output': 0.5, 'domain_output': 1.0}, optimizer=optimizer, metrics=['accuracy'])
        self.class_clf.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=optimizer, metrics=['accuracy'])
        self.domain_clf.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=optimizer, metrics=['accuracy'])
        self.embeddings_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=optimizer, metrics=['accuracy'])

    def __batch_generator(self, X, y=None):
        ind = np.arange(X.shape[0])

        while True:
            np.random.shuffle(ind)
            mini_batch_indices = np.random.choice(ind, size=self.batch_size // 2, replace=False)
            if y is not None:
                yield X[mini_batch_indices], y[mini_batch_indices]
            else:
                yield X[mini_batch_indices]

    def fit(self, Xs, ys, Xt, yt, epochs=1):
        batches_per_epoch = Xs.shape[0] // self.batch_size
        source_batches = self.__batch_generator(Xs, ys)
        target_batches = self.__batch_generator(Xt, yt)

        print('|{:^25}|{:^25}|{:^25}|{:^25}|{:^25}|{:^25}|'.format('epoch', 'loss', 'closs', 'dloss', 'source acc', 'target acc'))

        history = {'accs': [], 'acct': [], 'loss': [], 'closs': [], 'dloss': []}
        for i in range(epochs):
            accs = acct = loss = closs = dloss = 0.0
            for _ in range(batches_per_epoch):
                Xsb, ysb = next(source_batches)
                Xtb, ytb = next(target_batches)
                
                allX = np.concatenate([Xsb, Xtb], axis=0)
                Ysb = tf.keras.utils.to_categorical(np.concatenate([ysb, ysb]),
                                                    num_classes=self.n_classes)
                domain_labels = tf.keras.utils.to_categorical(np.array([0] * Xsb.shape[0] + [1] * Xtb.shape[0]),
                                                              num_classes=2)

                stats = self.dann_model.train_on_batch(allX, {'classifier_output': Ysb, 'domain_output': domain_labels})

                Ysp = self.class_clf.predict(Xsb)
                accs += accuracy_score(ysb, Ysp.argmax(axis=1)) / batches_per_epoch

                Ytp = self.class_clf.predict(Xtb)
                acct += accuracy_score(ytb, Ytp.argmax(axis=1))  / batches_per_epoch

                loss += stats[0] / batches_per_epoch
                closs += stats[1] / batches_per_epoch
                dloss += stats[2] / batches_per_epoch
            
            history['accs'].append(accs)
            history['acct'].append(acct)
            history['loss'].append(loss)
            history['closs'].append(closs)
            history['dloss'].append(dloss)

            print('|{:^25}|{:^25}|{:^25}|{:^25}|{:^25}|{:^25}|'.format(i, loss, closs, dloss, accs, acct))
        return history

    def predict(self, X):
        return self.dann_model.predict(X)

    def get_representation(self, X):
        return self.embeddings_model.predict(X)