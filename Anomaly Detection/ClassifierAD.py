import Split
import numpy as np
import pandas as pd
from sklearn import model_selection, datasets
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn import ensemble
from sklearn import svm
from sklearn import decomposition
from sklearn.neighbors import NearestNeighbors


class cls:
    def __init__(self,df,x_train,x_test,y_train,y_test,X,y):
         
        self.df = df
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.X= X
        self.y = y

    def print_coeff_intrc(self, REG):
        self.reg = REG
        print("The coefficients fitted are : " , self.reg.coef_)
        print("The intercept fitted is : " , self.reg.intercept_)
    
    def RNT_IsoFor(self,*args,**kwargs):
        isofor = ensemble.IsolationForest(*args, **kwargs)    
        isofor.fit(self.x_train,self.y_train)
        y_pred_class = isofor.predict(self.x_test)
        #AccuracyAD.accu_scores(self.y_test, y_pred_class)
        #score = np.sqrt(-cross_val_score(isofor, self.X, self.y, cv=10, scoring='neg_mean_squared_error').mean())
        #print("Root Mean Squared Error = ",score)
        #self.print_coeff_intrc(linreg)
        #mean_squared_error(y_pred_class, self.y_test)
        
    def RNT_OCSVM(self,*args,**kwargs):
        ocsvm = svm.OneClassSVM(*args, **kwargs)    
        ocsvm.fit(self.x_train,self.y_train)
        y_pred_class = ocsvm.predict(self.x_test)
        #AccuracyAD.accu_scores(self.y_test, y_pred_class)
        print("The indices of the modeled support vectors are : ", ocsvm.support_)
        #attr = ['coef_','dual_coef_']
        #Attribute.attribute(ocsvm,attr)
    
    '''def RNT_LOFac(self,*args,**kwargs):
        lofac = NearestNeighbors(n_neighbors=1,*args, **kwargs)    
        lofac.fit(self.x_train,self.y_train)
        y_pred_class = lofac.predict(self.x_test)
        #AccuracyAD.accu_scores(self.y_test, y_pred_class)
        attr = ['negative_outlier_factor_ ','n_neighbors_']
        Attribute(lofac, attr)'''
    
    def RNT_PCA(self,*args,**kwargs):
        pca = decomposition.PCA(*args, **kwargs)    
        pca.fit(self.x_train,self.y_train)
        #y_pred_class = pca.predict(self.x_test)
        #AccuracyAD.accu_scores(self.y_test, y_pred_class)
        print("The principal components are : " , pca.components_)
        print("The percentage of explained variance by the PCA model is : " , pca.explained_variance_ratio_ * 100)
              
    
