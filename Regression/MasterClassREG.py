import Split
import numpy as np
import AccuracyREG
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import model_selection, datasets
import ClassifierReg
import os
from time import time
from sklearn.preprocessing import scale

class Master:

    def __init__(self,data):
        self.data = data
        self.readConverter()
 
    def readConverter(self):
    
        ext = os.path.splitext(self.data)[-1].lower()
        #print(ext)

        if ext == ".txt":
            self.df = pd.read_csv(self.data, sep=" ", header = None)
        elif ext == ".csv":
            self.df = pd.read_csv(self.data)
        elif ext == ".xlsx":
            self.df = pd.read_excel(open(self.data,'rb'), sheetname=0)
        elif ext == ".xls":
            self.df = pd.read_excel(open(self.data,'rb'), sheetname=0)

    def pass_features(self,featureCol):
        self.X = self.df[featureCol]
        self.X= scale(self.X)
       

    def pass_labels(self,labels):
        self.y = self.df[labels]
        


    def split_data(self,split):
        splitInst = Split.splitting(self.X, self.y, split)
        self.x_train,self.x_test,self.y_train,self.y_test = splitInst.get_train_test()

    def get_X(self):
        return self.X
    def get_y(self):
        return self.y
    
    def timer_start(self):
        self.start = time()
        return self.start
    
    def timer_stop(self, name):
        self.elapsed = time()
        self.elapsed = self.elapsed - self.start
        print("Time spent in", name, "is = ", self.elapsed)
        
    def classifierreg(self):
        self.reg = ClassifierReg.cls(self.df,self.x_train,self.x_test,self.y_train,self.y_test,self.X,self.y)

    '''LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
URL = http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html'''
    
    def LinReg(self, *args, **kwargs):
        self.timer_start()
        self.reg.RNT_LinReg(*args, **kwargs)
        self.timer_stop('Linear Regression')
    
    '''SVR(kernel='rbf', degree=3, gamma='auto', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1)'''
    '''URL = http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR'''
    
    def SVR(self, *args, **kwargs):
        self.timer_start()
        self.reg.RNT_SVR(*args, **kwargs)
        self.timer_stop('SV Regression')    
        
    '''Ridge(alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, solver='auto', random_state=None)[source]
URL = http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html'''
    
    def Ridge(self, *args, **kwargs):
        self.timer_start()
        self.reg.RNT_Ridge(*args, **kwargs)
        self.timer_stop('Ridge Regression')
        
    '''BayesianRidge(n_iter=300, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, compute_score=False, fit_intercept=True, normalize=False, copy_X=True, verbose=False)[source]
URL = http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html'''    
    
    def BayRidge(self, *args, **kwargs):
        self.timer_start()
        self.reg.RNT_BayRidge(*args, **kwargs)
        self.timer_stop('Bayesian Ridge Regression')
        
    '''DecisionTreeRegressor(criterion='mse', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_split=1e-07, presort=False)
URL = http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html'''
    
    def DecTreeReg(self, *args, **kwargs):
        self.timer_start()
        self.reg.RNT_DecTreeReg(*args, **kwargs)
        self.timer_stop('Decision Tree Regression')
        
    '''SGDRegressor(loss='squared_loss', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, n_iter=5, shuffle=True, verbose=0, epsilon=0.1, random_state=None, learning_rate='invscaling', eta0=0.01, power_t=0.25, warm_start=False, average=False)[source]¶
URL = http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html'''    
     
    def SGDReg(self, *args, **kwargs):
        self.timer_start()
        self.reg.RNT_SGDReg(*args, **kwargs)
        self.timer_stop('Stochastic Gradient Descent Regression')
        
    '''RandomForestRegressor(n_estimators=10, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False)
URL = http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html'''
    
    def RanForReg(self, *args, **kwargs):
        self.timer_start()
        self.reg.RNT_RanForReg(*args, **kwargs)
        self.timer_stop('Random Forest Regression')
        
    '''AdaBoostRegressor(base_estimator=None, n_estimators=50, learning_rate=1.0, loss='linear', random_state=None)
URL = http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html'''    
        
    def AdaBoostReg(self, *args, **kwargs):
        self.timer_start()
        self.reg.RNT_AdaBoostReg(*args, **kwargs)
        self.timer_stop('Adaptive Boost Regression')
        
    '''Lasso(alpha=1.0, fit_intercept=True, normalize=False, precompute=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic')
URL = http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html'''   
        
    def LassoReg(self, *args, **kwargs):
        self.timer_start()
        self.reg.RNT_LassoReg(*args, **kwargs)
        self.timer_stop('Lasso Regression')
        
        