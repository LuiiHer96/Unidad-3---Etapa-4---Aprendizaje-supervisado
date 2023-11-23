# -*- coding: utf-8 -*-

# Importar librerias
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt
import pandas as pd

# Cargar datos desde CSV
df = pd.read_csv('Cleaned-Data2.csv')

# Excluir columnas 'Severity_Severe' y 'Country'
X = df.drop(['Severity_Severe', 'Country'], axis=1)
y = df['Severity_Severe']

# Dividir datos en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenar modelo KNN con k = 5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predicciones en el conjunto de prueba
y_pred = knn.predict(X_test)

# Métricas de evaluación
print("Precisión:", accuracy_score(y_test, y_pred))
print("Reporte de clasificación:\n", classification_report(y_test, y_pred))
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))

# Curva de precisión vs K
k_range = range(1, 11)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred))
plt.plot(k_range, scores)
x_axis = list(k_range)
plt.xticks(x_axis)
plt.xlabel('Valor de K')
plt.ylabel('Precisión')
plt.title('Curva de Precisión vs. K')
plt.show()

# Buscar mejor K con GridSearchCV
knn = KNeighborsClassifier()
k_range = range(1, 11)
param_grid = dict(n_neighbors=k_range)
grid = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)
print("Mejor k:", grid.best_params_)

# Entrenar con el mejor k encontrado
best_k = grid.best_params_['n_neighbors']
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)

# Evaluación con el conjunto de prueba
y_pred = knn.predict(X_test)
print("Reporte de clasificación con el mejor k:\n", classification_report(y_test, y_pred))

# Métricas adicionales con el mejor k
precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average='weighted')
print("Precision: {}".format(precision))
print("Recall: {}".format(recall))
print("F1 Score: {}".format(fscore))