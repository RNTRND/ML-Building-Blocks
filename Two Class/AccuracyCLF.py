import numpy as np
from sklearn.svm import SVC
import pandas as pd
from sklearn import model_selection, datasets
from sklearn import metrics

class accu_scores:
    def __init__(self,y_test, y_pred_class):
        self.y_test = y_test
        self.y_pred_class = y_pred_class
        self.simpleAccu()
        self.confMat()
        self.recall()
        self.precision()
    def simpleAccu(self):
        print("simple accuracy : ",metrics.accuracy_score(self.y_test, self.y_pred_class))
    def confMat(self):
        print("confusion matrix : ", metrics.confusion_matrix(self.y_test, self.y_pred_class))
    def recall(self):
        print("recall score : ",metrics.recall_score(self.y_test, self.y_pred_class))
    def precision(self):
        print("precision score : ", metrics.precision_score(self.y_test, self.y_pred_class))