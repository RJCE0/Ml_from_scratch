import numpy as np

"""
Concept of Simple Linear Regression: 
A supervised learning algorithm that attempts to model the correlation amongst observed data 
and fitting a linear equation to it

process: 
1. using y = wx+b we will come up with a prediction result, y_hat
2. Calculate MSE between our predicition y_hat and actual label
3. Use gradient descent to decide our next weight and bias
4. Repeat until the gradient change is at some ε threshold to signfiy convergence
5. Now we will have a y = wx+b equation fit to our training set, and ready for test set

"""

class LinearRegression:
    
    def __init__(self, learning_rate = 0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.iterations):
            y_hat = np.dot(X, self.weights) + self.bias 
            
            dJdW = (1/(2*n_samples)) *  np.dot(X.T, (y_hat - y))
            dBdW = (1/(2*n_samples)) * np.sum(y_hat - y)
            
            self.weights = self.weights - (self.learning_rate * dJdW)
            self.bias = self.bias - (self.learning_rate * dBdW)
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias