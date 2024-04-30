import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score


def lr_search(X_train, y_train, X_test, y_test, std = True, printing = False):
    """
    Parameters
    ----------
    X_train, y_train, X_test, y_test : np.array or pd.DataFrame
        The dataset that you are working with
    std : Boolean
        Whether you have to standardize the data or not
    printing : Boolean
        Whether you want to print the results in python cells or not

    Returns
    -------
    score_lr, roc_auc_lr : float
        The accuracy and ROC AUC score of the model
    """
    #Use grid search for finding optimal hyperparameters for LR method
    if std:
        pipe_LR = make_pipeline(StandardScaler(),
                        LogisticRegression(random_state=1))
    else:
        pipe_LR = make_pipeline(LogisticRegression(random_state=1))
                        
    param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    solver_range=["newton-cg","lbfgs","liblinear"]

    param_grid = [{'logisticregression__C': param_range, 
                'logisticregression__solver': solver_range}]

    gs = GridSearchCV(estimator=pipe_LR, 
                    param_grid=param_grid, 
                    scoring='accuracy', 
                    refit=True,
                    cv=5,
                    n_jobs=-1)
    gs = gs.fit(X_train, y_train)
    clf = gs.best_estimator_
    score_lr = clf.score(X_test, y_test)
    roc_auc_lr = roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])

    if printing:
        print(gs.best_score_)
        print(gs.best_params_)
        print('Test accuracy: %.3f' % score_lr)

    return score_lr, roc_auc_lr


def svm_search(X_train, y_train, X_test, y_test, std = True, printing = False):
    """
    Parameters
    ----------
    X_train, y_train, X_test, y_test : np.array or pd.DataFrame
        The dataset that you are working with
    std : Boolean
        Whether you have to standardize the data or not
    printing : Boolean
        Whether you want to print the results in python cells or not

    Returns
    -------
    score_lr, roc_auc_lr : float
        The accuracy and ROC AUC score of the model
    """
    if std:
        pipe_svm = make_pipeline(StandardScaler(),
                        SVC(random_state=1,probability=True))
    else:
        pipe_svm = make_pipeline(SVC(random_state=1,probability=True))
                        
    param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    param_grid = [{'svc__C': param_range, 
                'svc__kernel': ['linear']},
                {'svc__C': param_range, 
                'svc__gamma': param_range, 
                'svc__kernel': ['rbf']}]

    gs = GridSearchCV(estimator=pipe_svm, 
                    param_grid=param_grid, 
                    scoring='accuracy', 
                    refit=True,
                    cv=5,
                    n_jobs=-1)
    gs = gs.fit(X_train, y_train)
    clf = gs.best_estimator_
    score_svm = clf.score(X_test, y_test)
    roc_auc_svm = roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])

    if printing:
        print(gs.best_score_)
        print(gs.best_params_)
        print('Test accuracy: %.3f' % score_svm)

    return score_svm, roc_auc_svm


def rf_search(X_train,y_train, X_test, y_test, std = True, printing = False):
    """
    Parameters
    ----------
    X_train, y_train, X_test, y_test : np.array or pd.DataFrame
        The dataset that you are working with
    std : Boolean
        Whether you have to standardize the data or not
    printing : Boolean
        Whether you want to print the results in python cells or not

    Returns
    -------
    score_lr, roc_auc_lr : float
        The accuracy and ROC AUC score of the model
    """
    if std:
        pipe_rf = make_pipeline(StandardScaler(),
                        RandomForestClassifier(random_state=1))
    else:
        pipe_rf = make_pipeline(RandomForestClassifier(random_state=1))
                        
    param_range = list(range(2, 11))

    param_grid = [{'randomforestclassifier__max_depth': param_range, 
                'randomforestclassifier__criterion': ['gini', 'entropy'],
                'randomforestclassifier__n_estimators': [25, 50, 100, 200, 300],
                'randomforestclassifier__bootstrap': [True, False]}]

    gs = GridSearchCV(estimator=pipe_rf, 
                    param_grid=param_grid, 
                    scoring='accuracy', 
                    refit=True,
                    cv=5,
                    n_jobs=-1)
    gs = gs.fit(X_train, y_train)
    clf = gs.best_estimator_
    score_rf = clf.score(X_test, y_test)
    roc_auc_rf = roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])

    if printing:
        print(gs.best_score_)
        print(gs.best_params_)
        print('Test accuracy: %.3f' % score_rf)

    return score_rf, roc_auc_rf

def multi_search(X_train, y_train, X_test, y_test, std = True, feature_type = 'name'):
    """
    Parameters
    ----------
    X_train, y_train, X_test, y_test : np.array or pd.DataFrame
        The dataset that you are working with
    std : Boolean
        Whether you have to standardize the data or not
    feature_type : str
        The method of feature selection

    Returns
    -------
    df: pd.DataFrame
        The accuracy and ROC AUC score of different models shown in a DataFrame
    """
    accuracy_lr, roc_auc_lr = lr_search(X_train, y_train, X_test, y_test, std)
    accuracy_svm, roc_auc_svm = svm_search(X_train, y_train, X_test, y_test, std)
    accuracy_rf, roc_auc_rf = rf_search(X_train, y_train, X_test, y_test, std)

    multi_index = pd.MultiIndex.from_product([[feature_type], ['accuracy', 'roc_auc']], names=['feature_type', 'score'])
    index = ['lr','svm','rf']
    data = np.array([[accuracy_lr, roc_auc_lr], [accuracy_svm, roc_auc_svm], [accuracy_rf, roc_auc_rf]])
    df = pd.DataFrame(data, index=index, columns=multi_index)

    return df
