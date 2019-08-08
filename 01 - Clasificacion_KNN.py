from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()

print(iris.keys())

x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target)

knn = KNeighborsClassifier(n_neighbors=2)

knn.fit(x_train, y_train)

print(knn.score(x_test, y_test))

print(iris.target_names[knn.predict([[1.2, 2.4, 5.6, 1.1]])])