import numpy as np
"""
    Concept of PCA:
    PCA can be seen as a subset of the concept of SVD, which is the generalisation of eigen decomposition. 
    
    Process: 
    1. Compute mean for every dimension of the dataset
    2. Compute Covariance matrix for whole dataset
    3. Compute eigenvectors and corresponding eigenvalues
    4. Sort the eigenvectors by decreasing eigenvalues and choose first k vectors depending on 
    how many dimensions you wish to reduce to. (Roughly speaking, the eigenvectors with 
    lowest eigenvalues bear the least information about the distribution of the data)
    5. Transform all the points by this new matrix formed of the first k vectors onto the new subspace

    Returns:
        The set of points X mapped onto the new subspace
"""

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