import Split
import numpy as np
import AccuracyCLF
import pandas as pd
from sklearn import model_selection, datasets
from sklearn.model_selection import cross_val_score
import ClassifierCLF
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
        print("Time spent in",name, "is = ", self.elapsed)
    
    def classifierclf(self):
        self.clf = ClassifierCLF.cls(self.df,self.x_train,self.x_test,self.y_train,self.y_test,self.X,self.y)
    
    def SVM(self, C=1.0, kernel = 'rbf',gamma = 'auto', *args, **kwargs):
        self.timer_start()
        self.clf.RNT_SVM(C, kernel, gamma, *args, **kwargs)
        self.timer_stop('Support Vector Machine')
    
    def NB(self):
        self.timer_start()
        self.clf.RNT_NB()
        self.timer_stop('Naive Bayes')
   
    def LogReg(self, C=1.0, solver='liblinear', *args, **kwargs):
        self.timer_start()
        self.clf.RNT_LogReg(C, solver, *args, **kwargs)
        self.timer_stop('Logistic Regression')
            
    def DecTree(self, min_samples_split = 2, *args, **kwargs):
        self.timer_start()
        self.clf.RNT_DecTree(min_samples_split, *args, **kwargs)
        self.timer_stop('Decision Tree')

    def KNN(self,n_neighbors=5, weights='uniform', *args, **kwargs):
        self.timer_start()
        self.clf.RNT_KNN(n_neighbors, weights, *args, **kwargs)
        self.timer_stop('K Nearest Neighbor')
        
    def AdaBoostClf(self,base_estimator = None, n_estimators=50, learning_rate=1.0, *args, **kwargs):
        self.timer_start()
        self.clf.RNT_AdaBoostClf(base_estimator, n_estimators, learning_rate, *args, **kwargs)
        self.timer_stop('AdaBoost')
    
    def RanForClf(self,max_depth=5, n_estimators=10, max_features=1, *args, **kwargs):
        self.timer_start()
        self.clf.RNT_RanForClf(max_depth, n_estimators,max_features=1, *args, **kwargs)
        self.timer_stop('Random Forest')



