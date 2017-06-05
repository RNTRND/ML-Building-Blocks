import Split
import numpy as np
import AccuracyCLF
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection, datasets
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier


class cls:
    def __init__(self,df,x_train,x_test,y_train,y_test,X,y):
         
        self.df = df
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.X= X
        self.y = y

    
    def RNT_SVM(self,C ,kernel, gamma,*args,**kwargs):
        svc = SVC(C=C,kernel = kernel, gamma = gamma, *args, **kwargs)    
        svc.fit(self.x_train,self.y_train)
        y_pred_class = svc.predict(self.x_test)
        AccuracyCLF.accu_scores(self.y_test, y_pred_class)
        print("cross validation score of roc auc",cross_val_score(svc, self.X, self.y, cv=10, scoring='roc_auc').mean())
       
        
    def RNT_NB(self):    
        clf = GaussianNB()
        clf.fit(self.x_train, self.y_train)
        y_pred_class = clf.predict(self.x_test)
        AccuracyCLF.accu_scores(self.y_test, y_pred_class)
        print("cross validation score of roc auc",cross_val_score(clf, self.X, self.y, cv=10, scoring='roc_auc').mean())    
        
    def RNT_LogReg(self, C, solver, *args, **kwargs): 
        clf = LogisticRegression(C = C, solver = solver, *args, **kwargs)
        clf.fit(self.x_train, self.y_train)
        y_pred_class = clf.predict(self.x_test)
        AccuracyCLF.accu_scores(self.y_test, y_pred_class)
        
    def RNT_KNN(self,n_neighbors, weights, *args, **kwargs):
        clf = KNeighborsClassifier(n_neighbors = n_neighbors, weights = weights)
        clf.fit(self.x_train, self.y_train)
        y_pred_class = clf.predict(self.x_test)
        AccuracyCLF.accu_scores(self.y_test, y_pred_class)
        print("cross validation score of roc auc",cross_val_score(clf, self.X, self.y, cv=10, scoring='roc_auc').mean())

    def RNT_DecTree(self, min_samples_split, *args, **kwargs):
        clf = DecisionTreeClassifier(min_samples_split = min_samples_split, *args, **kwargs)
        clf.fit(self.x_train, self.y_train)
        y_pred_class = clf.predict(self.x_test)
        AccuracyCLF.accu_scores(self.y_test, y_pred_class)
        print("cross validation score of roc auc",cross_val_score(clf, self.X, self.y, cv=10, scoring='roc_auc').mean())
           
    def RNT_AdaBoostClf(self,base_estimator, n_estimators, learning_rate, *args, **kwargs):
        clf = AdaBoostClassifier(base_estimator = base_estimator, n_estimators = n_estimators, learning_rate = learning_rate)
        clf.fit(self.x_train, self.y_train)
        y_pred_class = clf.predict(self.x_test)
        AccuracyCLF.accu_scores(self.y_test, y_pred_class)
        print("cross validation score of roc auc",cross_val_score(clf, self.X, self.y, cv=10, scoring='roc_auc').mean())
       
    def RNT_RanForClf(self,max_depth, n_estimators,max_features=1, *args, **kwargs):
        clf = RandomForestClassifier(max_depth = max_depth, n_estimators = n_estimators, max_features = max_features, *args, **kwargs)
        clf.fit(self.x_train, self.y_train)
        y_pred_class = clf.predict(self.x_test)
        AccuracyCLF.accu_scores(self.y_test, y_pred_class)
        print("cross validation score of roc auc",cross_val_score(clf, self.X, self.y, cv=10, scoring='roc_auc').mean()) 






