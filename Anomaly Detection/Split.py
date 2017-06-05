from sklearn import model_selection, datasets
from sklearn import metrics

class splitting:
    def __init__(self,X,y,split):
        
        self.x_train,self.x_test,self.y_train,self.y_test = model_selection.train_test_split(X,y,test_size = split)
      

    def get_train_test(self):
        
        return self.x_train,self.x_test,self.y_train,self.y_test
        
      
