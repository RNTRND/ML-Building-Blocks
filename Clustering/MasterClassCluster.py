import Split
import numpy as np
import AccuracyREG
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import model_selection, datasets
import Clustering
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
        
    
    
    def clustering(self):
        self.cluster = Clustering.cls(self.X,self.y)
        
    '''AffinityPropagation(damping=0.5, max_iter=200, convergence_iter=15, copy=True, preference=None, affinity='euclidean', verbose=False)'''
    '''URL = http://scikit-learn.org/stable/modules/generated/sklearn.cluster.AffinityPropagation.html'''
    
    def AffinityPro(self,*args,**kwargs):
        self.timer_start()
        self.cluster.RNT_AffinityPro(*args,**kwargs)
        self.timer_stop('Affinity Propagation')
        
    '''KMeans(n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1, algorithm='auto')
    URL = http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    '''
    
    
    def KMeans(self,*args,**kwargs):
        self.timer_start()
        self.cluster.RNT_KMeans(*args,**kwargs)
        self.timer_stop('K Means')
        
    
    '''SpectralClustering(n_clusters=8, eigen_solver=None, random_state=None, n_init=10, gamma=1.0, affinity='rbf', n_neighbors=10, eigen_tol=0.0, assign_labels='kmeans', degree=3, coef0=1, kernel_params=None, n_jobs=1)
URL = http://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html'''
    
    def SpectralCluster(self,*args,**kwargs):
        self.timer_start()
        self.cluster.RNT_SpectralCluster(*args,**kwargs)
        self.timer_stop('Spectral Clustering')     
            
    
    '''MeanShift(bandwidth=None, seeds=None, bin_seeding=False, min_bin_freq=1, cluster_all=True, n_jobs=1)

       URL= http://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html'''
    
    def MeanShift(self,*args,**kwargs):
        self.timer_start()
        self.cluster.RNT_MeanShift(*args,**kwargs)
        self.timer_stop('Mean SHift')
        
    '''MiniBatchKMeans(n_clusters=8, init='k-means++', max_iter=100, batch_size=100, verbose=0, compute_labels=True, random_state=None, tol=0.0, max_no_improvement=10, init_size=None, n_init=3, reassignment_ratio=0.01)

   URL = http://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html'''
    
    
    def BatchKMeans(self,*args,**kwargs):
        self.timer_start()
        self.cluster.RNT_BatchKMeans(*args,**kwargs)
        self.timer_stop('Batch K Means')
        
    
    '''GaussianMixture(n_components=1, covariance_type='full', tol=0.001, reg_covar=1e-06, max_iter=100, n_init=1, init_params='kmeans', weights_init=None, means_init=None, precisions_init=None, random_state=None, warm_start=False, verbose=0, verbose_interval=10)

   URL = http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture'''
    
    
    def GMM(self,*args,**kwargs):
        self.timer_start()
        self.cluster.RNT_GMM(*args,**kwargs)
        self.timer_stop('GMM')   
        
    
    '''BayesianGaussianMixture(n_components=1, covariance_type='full', tol=0.001, reg_covar=1e-06, max_iter=100, n_init=1, init_params='kmeans', weight_concentration_prior_type='dirichlet_process', weight_concentration_prior=None, mean_precision_prior=None, mean_prior=None, degrees_of_freedom_prior=None, covariance_prior=None, random_state=None, warm_start=False, verbose=0, verbose_interval=10)

    URL = http://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html#sklearn.mixture.BayesianGaussianMixture'''
    
    def VBGMM(self,*args,**kwargs):
        self.timer_start()
        self.cluster.RNT_VBGMM(*args,**kwargs)
        self.timer_stop('GMM')    
   

        