import numpy as np
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()

print(iris.keys())

X_train, X_test, Y_train, Y_test = train_test_split(iris['data'], iris['target'])

# Clasificador vecinos cercanos
knn = KNeighborsClassifier(n_neighbors=2)

# Entrenar
knn.fit(X_train, Y_train)

KNeighborsClassifier(algorithm = 'auto',
                     leaf_size = 30,
                     metric = 'minkowski',
                     metric_params = None,
                     n_jobs = 1,
                     n_neighbors = 7,
                     p = 2,
                     weights = 'uniform')

print (knn.score(X_test, Y_test))

print(iris['target_names'][knn.predict([[1.2, 3.4, 5.6, 1.1]])])
