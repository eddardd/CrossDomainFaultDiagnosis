import ot
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils import eval_clf
from utils import scatterplot
from utils import make_grid
from utils import plot_contours

from domain_adaptation.instance_based import KMM
from domain_adaptation.instance_based import KLIEP
from domain_adaptation.ot_based.jdot import hinge_loss

from domain_adaptation import OTClassifier
from domain_adaptation import JDOTClassifier
from domain_adaptation import WeightedClassifier

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams['mathtext.fontset'] = 'custom'  
plt.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'  
plt.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'  
plt.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'  
plt.rcParams['font.size'] = 16  
plt.rcParams['mathtext.fontset'] = 'stix'  
plt.rcParams['font.family'] = 'STIXGeneral' 

# Constants
SHOW_FIGURES = False

# Palette
palette = sns.color_palette('tab10', n_colors=3)
palette_cmap = ListedColormap(palette.as_hex())

# Classifier
reg = 1e-2
gamma = 0.1
kernel = 'linear'
clf = LinearSVC(penalty='l2', loss='squared_hinge', C=reg, max_iter=1e+4)

# Data Generation
ns = 300
nt = 300
nz = 0.3

Xs, ys = ot.datasets.get_data_classif('3gauss', ns, nz)
Xt, yt = ot.datasets.get_data_classif('3gauss', nt, nz)

accs = []
algorithms = []
angles = []
df = pd.DataFrame()

for theta in np.arange(.05 * np.pi, np.pi + .05 * np.pi, .05 * np.pi):
    rotation = np.array([[np.cos(theta),np.sin(theta)],
                        [-np.sin(theta),np.cos(theta)]])

    Xt = np.dot(Xt, rotation.T)

    print('|' + '-' * 51 + '|')
    print('|{:^51}|'.format('Angle = {}'.format(theta * 180 / np.pi)))
    print('|' + '-' * 51 + '|')
    print('|{:^25}|{:^25}|'.format('Algorithm', 'Accuracy'))
    print('|' + '-' * 51 + '|')

    # Baseline
    clf.fit(Xs, ys)
    yp = clf.predict(Xt)
    acc_baseline = accuracy_score(yt, yp)
    print('|{:^25}|{:^25}|'.format('Baseline', acc_baseline))
    angles.append(theta)
    accs.append(acc_baseline)
    algorithms.append('Baseline')

    if SHOW_FIGURES:
        fig, ax = plt.subplots(1, 1, figsize=(9, 5))

        clf, acc_baseline, cmat = eval_clf(clf, Xs, ys, Xt, yt)
        xx, yy = make_grid(x=np.concatenate([Xs, Xt], axis=0)[:, 0],
                        y=np.concatenate([Xs, Xt], axis=0)[:, 1])
        plot_contours(ax, clf, xx, yy, edgecolors='face', alpha=.1, cmap=palette_cmap, shading='auto')

        ls = np.array([r'$C_{' + str(int(t + 1)) + r'}$' for t in ys])
        lt = np.array([r'$C_{' + str(int(t + 1)) + r'}$' for t in yt])

        scatterplot(Xs, label_names=ls, ax=ax, column_names=[r'x', r'y'])
        scatterplot(Xt, label_names=lt, ax=ax, marker='x', column_names=[r'x', r'y'])
        ax.legend().remove()
        plt.show()


    # KLIEP
    kliep_clf = WeightedClassifier(clf=clf, importance_estimator=KLIEP, 
                                kernel='rbf', numItermax=1000, epsilon=1e-4,
                                sigma=.25, support_vectors=Xs)
    kliep_clf.fit(Xs, ys, Xt)
    yp = kliep_clf.predict(Xt)
    kliep_acc = accuracy_score(yt, yp)
    print('|{:^25}|{:^25}|'.format('KLIEP', kliep_acc))
    angles.append(theta)
    accs.append(kliep_acc)
    algorithms.append('KLIEP')

    if SHOW_FIGURES:
        fig = plt.figure(figsize=(9, 5))
        plt.scatter(Xs[:, 0], Xs[:, 1], c=kliep_clf.weights, cmap='jet', vmin=0.0, vmax=np.max(kliep_clf.weights))
        im = plt.scatter(Xs[:, 0], Xs[:, 1], c=kliep_clf.weights, cmap='jet', vmin=0.0, vmax=np.max(kliep_clf.weights))
        plt.colorbar(im)

        fig, ax = plt.subplots(1, 1, figsize=(9, 5))
        xx, yy = make_grid(x=np.concatenate([Xs, Xt], axis=0)[:, 0],
                        y=np.concatenate([Xs, Xt], axis=0)[:, 1])
        plot_contours(ax, clf, xx, yy, edgecolors='face', alpha=.05, cmap=palette_cmap, shading='auto')

        ls = np.array([r'$C_{' + str(int(t + 1)) + r'}$' for t in ys])
        lt = np.array([r'$C_{' + str(int(t + 1)) + r'}$' for t in yt])

        scatterplot(Xs, label_names=ls, ax=ax, column_names=[r'x', r'y'])
        scatterplot(Xt, label_names=lt, ax=ax, marker='x', column_names=[r'x', r'y'])
        ax.legend().remove()
        plt.show()

    # KMM
    kmm_clf = WeightedClassifier(clf=clf, importance_estimator=KMM, 
                                kernel='rbf', B=1000, gamma=1)
    kmm_clf.fit(Xs, ys, Xt)
    yp = kmm_clf.predict(Xt)
    kmm_acc = accuracy_score(yt, yp)
    print('|{:^25}|{:^25}|'.format('KMM', kmm_acc))
    angles.append(theta)
    accs.append(kmm_acc)
    algorithms.append('KMM')

    if SHOW_FIGURES:
        fig = plt.figure(figsize=(9, 5))
        plt.scatter(Xs[:, 0], Xs[:, 1], c=kmm_clf.weights, cmap='jet', vmin=0.0, vmax=np.max(kmm_clf.weights))
        im = plt.scatter(Xs[:, 0], Xs[:, 1], c=kmm_clf.weights, cmap='jet', vmin=0.0, vmax=np.max(kmm_clf.weights))
        plt.colorbar(im)

        fig, ax = plt.subplots(1, 1, figsize=(9, 5))
        xx, yy = make_grid(x=np.concatenate([Xs, Xt], axis=0)[:, 0],
                        y=np.concatenate([Xs, Xt], axis=0)[:, 1])
        plot_contours(ax, clf, xx, yy, edgecolors='face', alpha=.05, cmap=palette_cmap, shading='auto')

        ls = np.array([r'$C_{' + str(int(t + 1)) + r'}$' for t in ys])
        lt = np.array([r'$C_{' + str(int(t + 1)) + r'}$' for t in yt])

        scatterplot(Xs, label_names=ls, ax=ax, column_names=[r'x', r'y'])
        scatterplot(Xt, label_names=lt, ax=ax, marker='x', column_names=[r'x', r'y'])
        ax.legend().remove()
        plt.show()

    # OT - Sinkhorn
    ot_clf = OTClassifier(clf=clf, ot_solver=ot.da.SinkhornTransport, reg_e=1e-2, norm='max')
    ot_clf.fit(Xs, ys, Xt, yt=None)
    yp = ot_clf.predict(Xt)
    acc = accuracy_score(yt, yp)
    print('|{:^25}|{:^25}|'.format('OT - Sinkhorn', acc))
    angles.append(theta)
    accs.append(acc)
    algorithms.append('Sinkhorn')

    if SHOW_FIGURES:
        fig, ax = plt.subplots(1, 1, figsize=(9, 5))
        xx, yy = make_grid(x=np.concatenate([ot_clf.tXs, Xt], axis=0)[:, 0],
                        y=np.concatenate([ot_clf.tXs, Xt], axis=0)[:, 1])
        plot_contours(ax, ot_clf.clf, xx, yy, edgecolors='face', alpha=.05, cmap=palette_cmap, shading='auto')

        ls = np.array([r'$C_{' + str(int(t + 1)) + r'}$' for t in ys])
        lt = np.array([r'$C_{' + str(int(t + 1)) + r'}$' for t in yt])

        scatterplot(ot_clf.tXs, label_names=ls, ax=ax, column_names=[r'x', r'y'])
        scatterplot(Xt, label_names=lt, ax=ax, marker='x', column_names=[r'x', r'y'])
        ax.legend().remove()
        plt.show()


    # OT - JDOT
    jdot_clf = JDOTClassifier(clf=clf, loss=hinge_loss, reg=1e-2, method='sinkhorn')
    jdot_clf.fit(Xs, ys, Xt, yt)
    yp = jdot_clf.predict(Xt)
    acc = accuracy_score(yt, yp)
    print('|{:^25}|{:^25}|'.format('JDOT', acc))
    print('|' + '-' * 51 + '|')
    angles.append(theta)
    accs.append(acc)
    algorithms.append('JDOT')

    if SHOW_FIGURES:
        fig, ax = plt.subplots(1, 1, figsize=(9, 5))
        xx, yy = make_grid(x=np.concatenate([Xs, Xt], axis=0)[:, 0],
                        y=np.concatenate([Xs, Xt], axis=0)[:, 1])
        plot_contours(ax, jdot_clf.clf, xx, yy, edgecolors='face', alpha=.05, cmap=palette_cmap, shading='auto')

        ls = np.array([r'$C_{' + str(int(t + 1)) + r'}$' for t in ys])
        lt = np.array([r'$C_{' + str(int(t + 1)) + r'}$' for t in yt])

        scatterplot(Xs, label_names=ls, ax=ax, column_names=[r'x', r'y'])
        scatterplot(Xt, label_names=lt, ax=ax, marker='x', column_names=[r'x', r'y'])
        ax.legend().remove()
        plt.show()

df['Angle'] = angles
df['Algorithm'] = algorithms
df['Accuracy'] = accs

df.to_csv('./data/ToyClf.csv')