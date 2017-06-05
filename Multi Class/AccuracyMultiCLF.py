
# coding: utf-8

# In[3]:



from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support as score




# In[6]:

class accu_scores:
    def __init__(self,y_test, y_pred_class):
        self.y_test = y_test
        self.y_pred_class = y_pred_class
        self.simpleAccu()
        self.confMat()
        
        precision, recall, fscore, support = score(y_test, y_pred_class)

        print('precision: {}'.format(precision))
        print('recall: {}'.format(recall))
        print('fscore: {}'.format(fscore))
        print('support: {}'.format(support))
    def simpleAccu(self):
        print("simple accuracy : ",metrics.accuracy_score(self.y_test, self.y_pred_class,))
    def confMat(self):
        print("confusion matrix : ", metrics.confusion_matrix(self.y_test, self.y_pred_class))

           


# In[ ]:



