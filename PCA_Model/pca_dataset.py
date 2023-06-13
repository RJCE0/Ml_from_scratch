import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from pca import PCA

iris_data = datasets.load_iris()
X, y = iris_data.data, iris_data.target

pca = PCA(2)
pca.fit(X)
X_projected = pca.transform(X)

print(f'[Before PCA] ~ No. Elements in X: {X.shape[0]}, Dimensionality of X: {X.shape[1]}')
print(f'[After PCA] ~ No. Elements in X: {X_projected.shape[0]}, Dimensionality of X: {X_projected.shape[1]}')

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

plt.scatter(x1, x2, c=y, edgecolors="none", alpha=0.8, cmap=plt.colormaps["viridis"])

plt.xlabel("Principle Component 1", fontsize=15)
plt.ylabel("Principle Component 2", fontsize=15)
plt.colorbar()
plt.show()