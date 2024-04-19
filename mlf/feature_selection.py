import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt

# The following class is for semi-auto feature selection

class SemiautoFeatureSelection():
    def __init__(self, T1, T2, T3, T4, classifiers):
        self.classifiers = classifiers
        self.T1 = T1
        self.T2 = T2
        self.T3 = T3
        self.T4 = T4

    def fit(self, X_train, X_test):
        self.classifiers_=[]

        obj_labels_q = X_train.select_dtypes(include=['object']).columns
        num_labels_q = X_train.select_dtypes(exclude=['object']).columns

        ohe=OneHotEncoder(categories='auto',drop='first')
        transf=ColumnTransformer([('onehot',ohe,obj_labels_q),
                          ('nothing','passthrough',num_labels_q)]
                         )
        
        self.X_train_q=transf.fit_transform(X_train).astype(float)
        self.X_test_q=transf.transform(X_test).astype(float)

        sc = StandardScaler()
        self.X_train_q = sc.fit_transform(self.X_train_q)
        self.X_test_q = sc.transform(self.X_test_q)

        return self

    def fit_one(self,X_train,X_test):
        self.classifiers_one=[]

        if X_train.dtypes=='object':
            X_train_qi=X_train.values
            X_test_qi=X_test.values
            ohe=OneHotEncoder(categories='auto',drop='first')
            self.X_train_qi=ohe.fit_transform(X_train_qi.reshape(-1,1)).toarray()
            self.X_test_qi=ohe.transform(X_test_qi.reshape(-1,1)).toarray()

            sc = StandardScaler()
            self.X_train_qi = sc.fit_transform(self.X_train_qi)
            self.X_test_qi = sc.transform(self.X_test_qi)

        else:
            self.X_train_qi=X_train.values.reshape(-1,1)
            self.X_test_qi=X_test.values.reshape(-1,1)

            sc = StandardScaler()
            self.X_train_qi = sc.fit_transform(self.X_train_qi)
            self.X_test_qi = sc.transform(self.X_test_qi)

        return self

    def get_score(self, y_train, y_test,clf):

        fitted_clf = clone(clf).fit(self.X_train_q,y_train)

        # bag = BaggingClassifier(estimator=fitted_clf,
        #                 n_estimators=500, 
        #                 max_samples=1.0, 
        #                 max_features=1.0, 
        #                 bootstrap=True, 
        #                 bootstrap_features=False, 
        #                 n_jobs=1, 
        #                 random_state=1)

        probs_q=fitted_clf.predict_proba(self.X_test_q)[:,1]
        auc_q=roc_auc_score(y_test,probs_q)

        return auc_q

    def get_score_one(self, y_train, y_test,clf):

        fitted_clf = clone(clf).fit(self.X_train_qi,y_train)

        # bag = BaggingClassifier(estimator=fitted_clf,
        #                 n_estimators=500, 
        #                 max_samples=1.0, 
        #                 max_features=1.0, 
        #                 bootstrap=True, 
        #                 bootstrap_features=False, 
        #                 n_jobs=1, 
        #                 random_state=1)

        probs_qi=fitted_clf.predict_proba(self.X_test_qi)[:,1]
        auc_qi=roc_auc_score(y_test,probs_qi)

        return auc_qi
    
    def comp_score(self,X_train,y_train,X_test,y_test,label):
        
        
        
        self.clf_auc_q=[]
        self.clf_auc_qi=[]
        for clf in self.classifiers:

            auc_q_=[]
            auc_=[]

            for q in range(len(label)):
                self.fit(X_train[label[q]],X_test[label[q]])
                auc_q_.append(self.get_score(y_train,y_test,clf))

                auc_qi_=[]

                for i in range(len(label[q])):
                    self.fit_one(X_train[label[q][i]],X_test[label[q][i]])
                    auc_qi_.append(self.get_score_one(y_train,y_test,clf))
                    
                auc_.append(auc_qi_)

            self.clf_auc_q.append(auc_q_)
            self.clf_auc_qi.append(auc_)

        return self
    
    def first_select(self,label):

        clf_label={}
        self.select_auc_q={}
        self.select_auc_qi={}

        for i,clf in enumerate(self.classifiers):
            label_select=[]

            select_q=[]
            select_qi=[]

            clf_auc=self.clf_auc_q[i]
            clf_auc_=self.clf_auc_qi[i]
            #print(clf_auc)
            for q in range(len(label)):

                select_qi_=[]

                clf_auc_q=np.array(clf_auc_[q])

                if clf_auc[q]>self.T2 and clf_auc_q.max()>self.T1:
                    label_select.append(label[q])
                    select_q.append(clf_auc[q])
                    select_qi_.append(clf_auc_q)

                select_qi.append(select_qi_)
                continue

            select_qi=[arr for arr in select_qi if arr!=[]]

            self.select_auc_q[clf]=select_q
            self.select_auc_qi[clf]=select_qi
            clf_label[clf]=label_select

        return clf_label

                
    def second_select(self,X,y,label):

        needed_label={}

        for idx,clf in enumerate(self.classifiers):
            needed_label[clf]=[]
            selected_label=label[clf]
            selected_q=self.select_auc_q[clf]
            selected_qi=self.select_auc_qi[clf]

            for i in range(len(selected_label)):

                if np.array(selected_qi[i]).max()>selected_q[i] or selected_q[i]-np.array(selected_qi[i]).max()<self.T3:

                    m=np.argmax(np.array(selected_qi[i]))

                    needed_label[clf].append(selected_label[i][m])

                else:
                    label_q=selected_label[i]

                    X_label_q=X.loc[:,label_q]
                    X_label_q['y']=y
                    correlation=X_label_q.corr()['y']
                    correlation=correlation.sort_values(ascending=False)
                    correlation=correlation.drop('y')

                    for i in range(len(correlation)):
                        label_=correlation.index[:i+1].tolist()

                        clf_=RandomForestClassifier(n_estimators=100,random_state=1)
                        clf_.fit(X_label_q[label_],y)

                        importance=clf_.feature_importances_

                        total_importance=importance.sum()
                        if total_importance>self.T4:
                            needed_label[clf]+=label_
                            # print(label_)
                            break

        return needed_label
    
    def get(self):
        return self.select_auc_q,self.select_auc_qi
    

# The following code is for conventional L1 LR feature selections

class LogisticFeatureSelection:
    def __init__(self, X_train, y_train, X_columns):
        self.X_train = X_train
        self.y_train = y_train
        self.X_columns = X_columns

    def lr_fit(self, c_range = (-5, 5)):
        weights, params, n_nonzero, features = [], [], [], []

        sc = StandardScaler()
        X_train_std = sc.fit_transform(self.X_train)
        
        c_min, c_max = c_range
        for c in np.arange(c_min, c_max):
            lr = LogisticRegression(C = 10.**c, 
                                    random_state = 1,
                                    penalty = 'l1', 
                                    solver = 'liblinear')
            lr.fit(X_train_std, self.y_train)
            coef = lr.coef_[0]
            weights.append(coef)
            params.append(10.**c)
            n_nonzero.append(np.count_nonzero(coef))
            features.append(self.X_columns[coef.nonzero()])

        weights = np.array(weights)
        self.weights = weights
        self.params = params
        self.n_nonzero = np.array(n_nonzero)
        self.features = features

        return weights, params

    def lr_plot_feature(self, xlim = (10e-6, None)):
        for i in range(self.weights.shape[1]):
            plt.plot(self.params, self.weights[:, i], label = self.X_columns[i])
        plt.ylabel('Weight Coefficient')
        plt.xlabel('C')
        plt.xlim(xlim)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xscale('log')
        plt.show()
        pass

    def lr_select_feature(self, less_than = 10):
        sorted_indices = np.argsort(self.n_nonzero)
        sorted_n_nonzero = self.n_nonzero[sorted_indices]
        index = np.searchsorted(sorted_n_nonzero, less_than, side='right') - 1
        feature_column = self.features[index]
        return feature_column
