import numpy as np

"""
Concept of K-means:
This is an unsupervised learning algorithm that separates the data into k clusters.
Each data sample is assigned to nearest centroids and centroid locations are then 
updated to the mean of the respective points iteratively

process: 
1. Initialise the K centroids locations randomly 
2. Assign points to the nearest centroids
3. Update the K centroid's location to be the mean of each respective cluster
4. Repeat until convergence is reached
"""

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

class Kmeans:
    
    def __init__(self, K=5, max_iterations=100):
        self.K = K
        self.max_iterations = max_iterations
        self.clusters = [[] for _ in range(self.K)]
        self.centroids = []
        
    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape
        
        rnd_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        #randomly assign centroid initial position to be at one of the points we have
        self.centroids = [self.X[idx] for idx in rnd_sample_idxs]
        
        for _ in range(self.max_iterations):
            self.clusters = self._generate_clusters(self.centroids)
            centroids_prev = self.centroids
            self.centroids = self._generate_centroids(self.clusters)
            
            if self._is_converged(centroids_prev, self.centroids):
                break
            
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(self.clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
                
        return labels
        
    def _generate_clusters(self, centroids):
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            distances = [euclidean_distance(sample, point) for point in centroids]
            centroid_idx = np.argmin(distances)
            clusters[centroid_idx].append(idx)
        return clusters
    
    def _generate_centroids(self, clusters):
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids
    
    def _is_converged(self, centroids_prev, centroids): 
        distances = [euclidean_distance(centroids_prev[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0
    
    