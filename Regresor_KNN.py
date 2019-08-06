###
### Precios de casas en boston
###

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor

boston = load_boston()

# Visualiza el nombre de los valores de los datos.
print(boston.keys())

# Existen 506 casas con 13 características.
print(boston.data.shape)

X_ent, X_test, Y_ent, Y_test = train_test_split(boston.data, boston.target)

knn=KNeighborsRegressor(n_neighbors=4)

# Entrenar
knn.fit(X_ent, Y_ent)

# Para ver que tal ha aprendido el algoritmo.
print('Aprendizaje de vecinos cercanos: %d', knn.score(X_test, Y_test))

rl=LinearRegression()
rl.fit(X_ent, Y_ent)

print ('Aprendizaja Lineal: %d', rl.score(X_test, Y_test))

ridge=Ridge()
ridge.fit(X_ent, Y_ent)

print ('Aprendizaje Ridge: %d', ridge.score(X_test, Y_test))


