import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

mlf_colors = ['#63b2ee', '#76da91', '#f8cb7f', '#f89588', '#7cd6cf', '#9192ab', '#7898e1', '#efa666', '#eddd86', '#9987ce', '#63b2ee', '#76da91']

def plot_roc(model, X, y, ax = None, title = 'Receiver Operating Characteristic (ROC) Curve'):
    probs = model.predict_proba(X)[:, 1]

    fpr, tpr, thresholds = roc_curve(y, probs)
    roc_auc = roc_auc_score(y, probs)

    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color=mlf_colors[3], lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], color=mlf_colors[0], lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    pass

def plot_score_by_sample_size(train_sizes, train_scores, test_scores, ax = None, ylim = [0.8, 1.03]):
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(train_sizes, train_mean,
            color = mlf_colors[0], marker = 'o',
            markersize = 5, label = 'Training accuracy')
    ax.fill_between(train_sizes,
                    train_mean + train_std,
                    train_mean - train_std,
                    alpha = 0.15, color = mlf_colors[0])
    ax.plot(train_sizes, test_mean,
            color = mlf_colors[1], linestyle = '--',
            marker = 's', markersize = 5,
            label = 'Validation accuracy')
    ax.fill_between(train_sizes,
                    test_mean + test_std,
                    test_mean - test_std,
                    alpha = 0.15, color = mlf_colors[1])

    ax.grid(True)
    ax.set_xlabel('Number of training examples')
    ax.set_ylabel('Accuracy')
    ax.legend(loc='lower right')
    ax.set_ylim(ylim)

    pass

def plot_score_by_parameters(param_range, train_scores, test_scores, ax = None, ylim = [0.8, 1.0]):
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(param_range, train_mean,
            color = mlf_colors[0], marker = 'o',
            markersize = 5, label = 'Training accuracy')
    ax.fill_between(param_range,
                    train_mean + train_std,
                    train_mean - train_std,
                    alpha = 0.15, color = mlf_colors[0])
    ax.plot(param_range, test_mean,
            color = mlf_colors[1], linestyle = '--',
            marker = 's', markersize = 5,
            label = 'Validation accuracy')
    ax.fill_between(param_range,
                    test_mean + test_std,
                    test_mean - test_std,
                    alpha = 0.15, color = mlf_colors[1])
    ax.grid(True)
    ax.set_xscale('log')
    ax.set_xlabel('Parameter C')
    ax.set_ylabel('Accuracy')
    ax.legend(loc='lower right')
    ax.set_ylim(ylim)

    pass