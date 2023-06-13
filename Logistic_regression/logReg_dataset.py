import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from logReg import LogisticRegression

data = datasets.load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1234, test_size=0.2)

clf = LogisticRegression()
clf.fit(X_train, y_train)
y_hat = clf.predict(X_test)

accuracy = np.sum(y_hat == y_test) / len(y_test)
print(accuracy)