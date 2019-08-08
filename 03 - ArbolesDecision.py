from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
import graphviz
import matplotlib.pyplot as plt
import numpy as np

iris = load_iris()
print (iris.keys())

X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target)

arbol = DecisionTreeClassifier(max_depth=1)

arbol.fit(X_train, Y_train)

# Comprobar que tal aprende el modelo, con un 0.94 está bastante bien.
print(arbol.score(X_test, Y_test))

export_graphviz(arbol, out_file='arbol.dot', class_names=iris.target_names,
                feature_names=iris.feature_names, impurity=False, filled=True)

with open('arbol.dot') as f:
    dot_graph = f.read()

graphviz.Source(dot_graph)

caract = iris.data.shape[1]
plt.barh(range(caract), arbol.feature_importances_)
plt.yticks(np.arange(caract), iris.feature_names)
plt.xlabel('Importancia de las características')
plt.ylabel('Características')
plt.show()

arbol = DecisionTreeClassifier(max_depth=2)

arbol.fit(X_train, Y_train)

print(arbol.score(X_test, Y_test))

# Comprobar que tal ha aprendido el modelo comprobando los datos que se han usado para el entrenamiento.
# Muestra un valor de 1.0, no es recomendable ya que se pueden producir sobre ajuste Machine Learning
print (arbol.score(X_train, Y_train))

n_classes = 3
plot_colors = "bry"
plot_step = 0.2

for pairidx, pair in enumerate ([[0, 1], [0, 2], [0, 3],
                                 [1, 2], [1, 3], [2, 3]]):
    X = iris.data[:, pair]
    Y = iris.target

    # Entrenar el algoritmo
    clf = DecisionTreeClassifier(max_depth=2).fit(X, Y)

    plt.subplot(2, 3, pairidx + 1)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contour(xx, yy, Z, cmap=plt.cm.Paired)

    plt.xlabel(iris.feature_names[pair[0]])
    plt.ylabel(iris.feature_names[pair[1]])

    # Dibujar los puntos de entrenamiento
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(Y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c = color, label = iris.target_names[i], cmap=plt.cm.Paired)

    plt.axis("tight")

plt.suptitle("Ejemplos de clasificador de Arboles de decisión")
plt.legend()
plt.show()
