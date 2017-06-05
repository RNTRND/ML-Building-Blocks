import Split
import numpy as np
import AccuracyREG
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import model_selection, datasets
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn import ensemble
from sklearn.svm import SVR

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
    
    def RNT_LinReg(self,*args,**kwargs):
        linreg = linear_model.LinearRegression(*args, **kwargs)    
        linreg.fit(self.x_train,self.y_train)
        y_pred_class = linreg.predict(self.x_test)
        
        
        
        AccuracyREG.accu_scores(self.y_test, y_pred_class)
        score = np.sqrt(-cross_val_score(linreg, self.X, self.y, cv=10, scoring='neg_mean_squared_error').mean())
        print("Root Mean Squared Error = ",score)
        self.print_coeff_intrc(linreg)
        #mean_squared_error(y_pred_class, self.y_test)
        
    def RNT_SVR(self,*args,**kwargs):
        svr = SVR(*args,**kwargs)
        svr.fit(self.x_train,self.y_train)
        y_pred_class = svr.predict(self.x_test)
        AccuracyREG.accu_scores(self.y_test, y_pred_class)
        score = np.sqrt(-cross_val_score(svr, self.X, self.y, cv=10, scoring='neg_mean_squared_error').mean())
        print("Root Mean Squared Error = ",score)
        #self.print_coeff_intrc(svr)
        
    
     
            
       
        
    
    def RNT_Ridge(self,*args,**kwargs):
        rireg = linear_model.Ridge(*args, **kwargs)    
        rireg.fit(self.x_train,self.y_train)
        y_pred_class = rireg.predict(self.x_test)
        AccuracyREG.accu_scores(self.y_test, y_pred_class)
        score = np.sqrt(-cross_val_score(rireg, self.X, self.y, cv=10, scoring='neg_mean_squared_error').mean())
        print("Root Mean Squared Error = ",score)
        self.print_coeff_intrc(rireg)
        
       
        
    def RNT_BayRidge(self,*args,**kwargs):
        brreg = linear_model.BayesianRidge(*args, **kwargs)    
        brreg.fit(self.x_train,self.y_train)
        y_pred_class = brreg.predict(self.x_test)
        AccuracyREG.accu_scores(self.y_test, y_pred_class)
        score = np.sqrt(-cross_val_score(brreg, self.X, self.y, cv=10, scoring='neg_mean_squared_error').mean())
        print("Root Mean Squared Error = ",score)
        self.print_coeff_intrc(brreg)
        
    def RNT_DecTreeReg(self,*args,**kwargs):
        dtreg = DecisionTreeRegressor(*args, **kwargs)    
        dtreg.fit(self.x_train,self.y_train)
        y_pred_class = dtreg.predict(self.x_test)
        AccuracyREG.accu_scores(self.y_test, y_pred_class)
        score = np.sqrt(-cross_val_score(dtreg, self.X, self.y, cv=10, scoring='neg_mean_squared_error').mean())
        print("Root Mean Squared Error = ",score)
        print("Feature importance matrix = " , dtreg.feature_importances_)

    def RNT_SGDReg(self,*args,**kwargs):
        sgdreg = linear_model.SGDRegressor(*args, **kwargs)    
        sgdreg.fit(self.x_train,self.y_train)
        y_pred_class = sgdreg.predict(self.x_test)
        AccuracyREG.accu_scores(self.y_test, y_pred_class)
        score = np.sqrt(-cross_val_score(sgdreg, self.X, self.y, cv=10, scoring='neg_mean_squared_error').mean())
        print("Root Mean Squared Error = ",score)
        self.print_coeff_intrc(sgdreg)
        
    def RNT_RanForReg(self,*args,**kwargs):
        rfreg = ensemble.RandomForestRegressor(*args, **kwargs)    
        rfreg.fit(self.x_train,self.y_train)
        y_pred_class = rfreg.predict(self.x_test)
        AccuracyREG.accu_scores(self.y_test, y_pred_class)
        score = np.sqrt(-cross_val_score(rfreg, self.X, self.y, cv=10, scoring='neg_mean_squared_error').mean())
        print("Root Mean Squared Error = ",score)
        print("Feature importance matrix = " , rfreg.feature_importances_)
        
    def RNT_AdaBoostReg(self,*args,**kwargs):
        abreg = ensemble.AdaBoostRegressor(*args, **kwargs)    
        abreg.fit(self.x_train,self.y_train)
        y_pred_class = abreg.predict(self.x_test)
        AccuracyREG.accu_scores(self.y_test, y_pred_class)
        score = np.sqrt(-cross_val_score(abreg, self.X, self.y, cv=10, scoring='neg_mean_squared_error').mean())
        print("Root Mean Squared Error = ",score)
        print("Feature importance matrix = " , abreg.feature_importances_)
        
    def RNT_LassoReg(self,*args,**kwargs):
        lreg = linear_model.Lasso(*args, **kwargs)    
        lreg.fit(self.x_train,self.y_train)
        y_pred_class = lreg.predict(self.x_test)
        AccuracyREG.accu_scores(self.y_test, y_pred_class)
        score = np.sqrt(-cross_val_score(lreg, self.X, self.y, cv=10, scoring='neg_mean_squared_error').mean())
        print("Root Mean Squared Error = ",score)
        self.print_coeff_intrc(lreg)