import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from Kmeans import Kmeans

X, y = make_blobs(centers=3, n_samples=500, n_features=2, shuffle=True, random_state=40)

clusters = len(np.unique(y))

k = Kmeans(K=clusters, max_iterations=150)
y_hat = k.predict(X)

fig, ax = plt.subplots(figsize=(12,8))

for i, idx in enumerate(k.clusters):
    point = X[idx].T
    ax.scatter(*point)
    
for point in k.centroids:
    ax.scatter(*point, marker="x", color="black", linewidth=2)

plt.show()