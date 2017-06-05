from sklearn import metrics
import numpy as np
import AccuracyREG
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import SpectralClustering
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture


class cls:
    def __init__(self,X,y):
         
        
        
        self.X= X
        self.y=y
        
    def RNT_AffinityPro(self,*args,**kwargs):
        af = AffinityPropagation(*args,**kwargs).fit(self.X)
        cluster_centers_indices = af.cluster_centers_indices_
        labels = af.labels_
        labels_true = self.y
        n_clusters_ = len(cluster_centers_indices)

        print('Estimated number of clusters: %d' % n_clusters_)
        print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
        print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
        print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
        print("Adjusted Rand Index: %0.3f"
              % metrics.adjusted_rand_score(labels_true, labels))
        print("Adjusted Mutual Information: %0.3f"
              % metrics.adjusted_mutual_info_score(labels_true, labels))
        print("Silhouette Coefficient: %0.3f"
              % metrics.silhouette_score(self.X, labels, metric='sqeuclidean'))
        
        
    def RNT_KMeans(self,*args,**kwargs):
        km = KMeans(*args,**kwargs).fit(self.X)
        
        labels = km.labels_
        print(labels)
        labels_true = self.y
        #n_clusters_ = len(cluster_centers_indices)

        #print('Estimated number of clusters: %d' % n_clusters_)
        print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
        print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
        print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
        print("Adjusted Rand Index: %0.3f"
              % metrics.adjusted_rand_score(labels_true, labels))
        print("Adjusted Mutual Information: %0.3f"
              % metrics.adjusted_mutual_info_score(labels_true, labels))
        print("Silhouette Coefficient: %0.3f"
              % metrics.silhouette_score(self.X, labels, metric='sqeuclidean'))   
        
    def RNT_SpectralCluster(self,*args,**kwargs):
        km = SpectralClustering(*args,**kwargs).fit(self.X)
        
        labels = km.labels_
        print(labels)
        labels_true = self.y
        #n_clusters_ = len(cluster_centers_indices)

        #print('Estimated number of clusters: %d' % n_clusters_)
        print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
        print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
        print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
        print("Adjusted Rand Index: %0.3f"
              % metrics.adjusted_rand_score(labels_true, labels))
        print("Adjusted Mutual Information: %0.3f"
              % metrics.adjusted_mutual_info_score(labels_true, labels))
        print("Silhouette Coefficient: %0.3f"
              % metrics.silhouette_score(self.X, labels, metric='sqeuclidean')) 
        
        

    def RNT_BatchKMeans(self,*args,**kwargs):
        km = MiniBatchKMeans(*args,**kwargs).fit(self.X)
        
        labels = km.labels_
        print(labels)
        labels_true = self.y
        #n_clusters_ = len(cluster_centers_indices)

        #print('Estimated number of clusters: %d' % n_clusters_)
        print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
        print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
        print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
        print("Adjusted Rand Index: %0.3f"
              % metrics.adjusted_rand_score(labels_true, labels))
        print("Adjusted Mutual Information: %0.3f"
              % metrics.adjusted_mutual_info_score(labels_true, labels))
        print("Silhouette Coefficient: %0.3f"
              % metrics.silhouette_score(self.X, labels, metric='sqeuclidean')) 
        
        
        
    def RNT_GMM(self,*args,**kwargs):
        km = GaussianMixture(*args,**kwargs).fit(self.X)
        
        labels = km.predict(self.X)
        print(labels)
        labels_true = self.y
        #n_clusters_ = len(cluster_centers_indices)

        #print('Estimated number of clusters: %d' % n_clusters_)
        print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
        print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
        print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
        print("Adjusted Rand Index: %0.3f"
              % metrics.adjusted_rand_score(labels_true, labels))
        print("Adjusted Mutual Information: %0.3f"
              % metrics.adjusted_mutual_info_score(labels_true, labels))
        print("Silhouette Coefficient: %0.3f"
              % metrics.silhouette_score(self.X, labels, metric='sqeuclidean'))   
        
        
    def RNT_VBGMM(self,*args,**kwargs):
        km = BayesianGaussianMixture(*args,**kwargs).fit(self.X)
        
        labels = km.predict(self.X)
        print(labels)
        labels_true = self.y
        #n_clusters_ = len(cluster_centers_indices)

        #print('Estimated number of clusters: %d' % n_clusters_)
        print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
        print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
        print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
        print("Adjusted Rand Index: %0.3f"
              % metrics.adjusted_rand_score(labels_true, labels))
        print("Adjusted Mutual Information: %0.3f"
              % metrics.adjusted_mutual_info_score(labels_true, labels))
        print("Silhouette Coefficient: %0.3f"
              % metrics.silhouette_score(self.X, labels, metric='sqeuclidean'))
        
        
    
    

        

