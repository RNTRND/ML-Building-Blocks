import Split
import numpy as np
import pandas as pd
from sklearn import model_selection, datasets
import ClassifierAD
import os
from time import time

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
        
    def classifierad(self):
        self.reg = ClassifierAD.cls(self.df,self.x_train,self.x_test,self.y_train,self.y_test,self.X,self.y)

    '''IsolationForest(n_estimators=100, max_samples='auto', contamination=0.1, max_features=1.0, bootstrap=False, n_jobs=1, random_state=None, verbose=0)[source]
URL = http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html#sklearn.ensemble.IsolationForest'''
    
    def IsoFor(self, *args, **kwargs):
        self.timer_start()
        self.reg.RNT_IsoFor(*args, **kwargs)
        self.timer_stop('Isolation Forest')
        
    '''OneClassSVM(kernel='rbf', degree=3, gamma='auto', coef0=0.0, tol=0.001, nu=0.5, shrinking=True, cache_size=200, verbose=False, max_iter=-1, random_state=None)
    URL = http://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html'''
    
    def OCSVM(self, *args, **kwargs):
        self.timer_start()
        self.reg.RNT_OCSVM(*args, **kwargs)
        self.timer_stop('One Class SVM')
        
    '''LocalOutlierFactor(n_neighbors=20, algorithm='auto', leaf_size=30, metric='minkowski', p=2, metric_params=None, contamination=0.1, n_jobs=1)
    URL = http://scikit-learn.org/dev/modules/generated/sklearn.neighbors.LocalOutlierFactor.html#sklearn.neighbors.LocalOutlierFactor'''
    
    def LOFac(self, *args, **kwargs):
        self.timer_start()
        self.reg.RNT_LOFac(*args, **kwargs)
        self.timer_stop('Local Outlier Factor')
            
    '''PCA(n_components=None, copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', random_state=None)
      URL = http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html'''
    
    def PCA(self, *args, **kwargs):
        self.timer_start()
        self.reg.RNT_PCA(*args, **kwargs)
        self.timer_stop('Principle Component Analysis')
    
        
    
        
        