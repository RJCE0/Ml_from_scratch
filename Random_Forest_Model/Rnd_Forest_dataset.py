from random import random
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from Rnd_Forest import RndForest

data = datasets.load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf = RndForest(n_trees=10)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

accuracy =  np.sum(y_test == predictions) / len(y_test)
print(f'The accuracy of this iteration\'s Random Forest model on the breast cancer dataset was {accuracy}')