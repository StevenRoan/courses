import numpy as np
from sklearn import *  # import datasets, linear_model
import sklearn as sl
import matplotlib.pyplot as plt

np.random.seed(0)
X, y = sl.datasets.make_moons(200, noise=0.20)
print (sl.__path__)
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
# Train the logistic rgeression classifier
clf = sl.linear_model.LogisticRegressionCV()
clf.fit(X, y)
# # Plot the decision boundary
print (sl.__path__)
plot_decision_boundary(lambda x: clf.predict(x))
plt.title("Logistic Regression")

plt.show()
