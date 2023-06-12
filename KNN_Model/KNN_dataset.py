import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from KNN import KNN

#Give it 3 colors
cmap = ListedColormap(['#00A86B', '#FF2400', '#1338BE'])
iris = datasets.load_iris()

X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1234, test_size=0.2)

plt.figure()
plt.scatter(X[:,2], X[:,3], c=y, cmap=cmap, edgecolors='k', s=20)
plt.show()

clf = KNN(k=5)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

# Statistical measures 
accuracy = np.sum(predictions == y_test) / len(y_test)
print(f'This is the accuracy of the KNN model with the iris data set: {accuracy}')
