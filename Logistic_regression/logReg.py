import numpy as np



class LogisticRegression:
    
    def __init__(self, learning_rate = 0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = 0
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        
        for _ in range(self.iterations):
            y_hat = 1 / (1 + np.exp(-np.dot(X, self.weights) + self.bias))
            
            dJdW = (1/(2*n_samples)) *  np.dot(X.T, (y_hat - y))
            dBdW = (1/(2*n_samples)) * np.sum(y_hat - y)
            
            self.weights = self.weights - (self.learning_rate * dJdW)
            self.bias = self.bias - (self.learning_rate * dBdW)
    
    def predict(self, X):
        y_hat = 1 / (1 + np.exp(-np.dot(X, self.weights) + self.bias))
        return [0 if y < 0.5 else 1 for y in y_hat]