import numpy as np


class PCA:
    def __init__(self, n_pcs):
        self.n_pcs = n_pcs
        self.pcs = None
        self.mean = None
    
    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        
        X = X - self.mean
        
        covariance = np.cov(X.T)
        eigenVectors, eigenValues = np.linalg.eig(covariance)
        
        eigenVectors = eigenVectors.T
        
        idxs = np.argsort(eigenValues)[::-1]
        eigenValues = eigenValues[idxs]
        eigenVectors = eigenVectors[idxs]
        
        self.pcs = eigenVectors[:self.n_pcs]
        
    
    def transform(self, X):
        X = X - self.mean
        return np.dot(X, self.pcs.T)