import numpy as np
from sklearn.svm import SVC
import pandas as pd
from sklearn import model_selection, datasets
from sklearn import metrics

class accu_scores:
    def __init__(self,y_test, y_pred_class):
        self.y_test = y_test
        self.y_pred_class = y_pred_class
        self.mean_absolute_error()
        self.mean_squared_error()
        self.median_absolute_error()
        self.r2_score()
    def mean_absolute_error(self):
        print("Mean Absolute Error : ",metrics.mean_absolute_error(self.y_test, self.y_pred_class))
    def mean_squared_error(self):
        print("Mean Squared Error : ", metrics.mean_squared_error(self.y_test, self.y_pred_class))
    def median_absolute_error(self):
        print("Median Absolute Error : ",metrics.median_absolute_error(self.y_test, self.y_pred_class))
    def r2_score(self):
        print("Regression Score Function : ", metrics.r2_score(self.y_test, self.y_pred_class))