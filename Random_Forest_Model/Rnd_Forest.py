import numpy as np
from collections import Counter


"""
Concept of Random Forest: 
A supervised, tree ensemble that alllows for classification and regression tasks to be carried out.
Similar to a bagged ensemble but solves the issue of having multiple trees with similar/same 
root node and subsquent splits   

process: 
1. Take a random k subset of features to split the dataset on
2. Calculate information gain with each possible split
3. Split the set by the feature with the highest information gain
4. Continue for all subsquent branchs until a stopping criteria is met.
5. Repeat steps 1-4 for as many decision trees we wish to have in our forest
6. Now finally we are left with a forest of trees whereby, 
   given a datapoint x will get the predictions for x from each tree.
   For a Classification - a majority vote decides the label
   For a Regression task - We take each tree's predicted value and calculate the mean 

"""

def _most_voted_label(y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
    def is_leaf_node(self):
        return self.value is not None
    
class DecisionTree:
    def __init__(self, max_depth=100, min_samples_split=2, n_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
    
    def _fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)
    
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))
        
        if (depth >= self.max_depth or n_labels==1 or n_samples<self.min_samples_split):
            leaf_val = _most_voted_label(y)
            return Node(value=leaf_val)
        
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
        
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        l_idxs, r_idxs = self._split(X[:, best_feature], best_thresh)
        left = self._grow_tree(X[l_idxs, :], y[l_idxs], depth+1)
        right = self._grow_tree(X[r_idxs, :], y[r_idxs], depth+1)
        return Node(best_feature, best_thresh, left, right)


    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = threshold

        return split_idx, split_threshold


    def _information_gain(self, y, X_column, threshold):
        parent_entropy = self._entropy(y)

        left_idxs, right_idxs = self._split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r

        information_gain = parent_entropy - child_entropy
        return information_gain

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p>0])

    def _predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
        


class RndForest:
    
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []
        
    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split, n_features=self.n_features)
            X_sample, y_sample = self._rnd_samples(X, y)
            tree._fit(X_sample, y_sample)
            self.trees.append(tree)
            
    def _rnd_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]
    
    def predict(self, X):
        predictions = np.array([tree._predict(X) for tree in self.trees])
        trees_preds = np.swapaxes(predictions, axis1=0, axis2=1)
        predictions = np.array([_most_voted_label(pred) for pred in trees_preds])
        return predictions