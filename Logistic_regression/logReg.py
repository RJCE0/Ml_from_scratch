import numpy as np

"""
Concept of Simple Linear Regression: 
A supervised learning algorithm that attempts to estimate the probablity of an observed datapoint
falling into one-of-two possible categories based on a given dataset of independent variables

process: 
1. using y =  1 / (1+e^-wx+b) we will come up with a prediction result, y_hat
2. Calculate Cross Entropy loss between our predicition y_hat and actual label
3. Use gradient descent to decide our next weight and bias
4. Repeat until the gradient change is  < ε threshold, signifying convergence or No. iterations reaches target
5. Now we will have a y =  1 / (1+e^-wx+b) probability equation (along with a probability threshold usually 0.5) 
   to illustate the predicted label for a new observed datapoint. 

"""

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