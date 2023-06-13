import numpy as np
from collections import Counter

"""
Concept of KNN: 
The process of k-nearest neighbour algorithm is a non-parametric, 
(algorithms that don't make strong assumptions about the form of the mapping function), 
supervised-learning (data is labelled) classifier. It uses the 'distance' to make 
classifications/predictions about how to predict a label for an individual datapoint

Process: 
1. Define a k, which is the number of neighbours to check (use odd number to avoid ties)
2. Define distance metric
3. Find the closest K datapoints to an individual datapoint
4. Select the most common label amongst the k neighbours to be the label for a classification task,
   for a regression task, we can use a weighted average from the k neighbours to predict the new value

"""

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

def manhattan_distance(x1, x2):
    return np.linalg.norm(x1-x2, ord=1)

def minkowski_distance(x1, x2, p):
    return np.sum(np.abs(x1-x2)**p, axis=0)**(1/p)

    
class KNN:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        self.X_train, self.y_train = X, y 
    
    
    def predict(self, X):
        predictions = [self.prediction(x) for x in X]
        return predictions
    
    def prediction(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        
        k_indexes = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indexes]
        
        most_voted_label = Counter(k_nearest_labels).most_common()
        
        return most_voted_label[0][0]