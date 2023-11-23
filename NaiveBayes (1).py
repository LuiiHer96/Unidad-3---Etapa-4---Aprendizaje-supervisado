# -*- coding: utf-8 -*-

# Importar librerias
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, roc_auc_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Cargar datos desde CSV
df = pd.read_csv('Cleaned-Data2.csv')

# Excluir columnas 'Severity_Severe' y 'Country'
X = df.drop(['Severity_Severe', 'Country'], axis=1)
y = df['Severity_Severe']

# Dividir datos en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Buscar mejores parámetros con GridSearchCV
params = {}
nb = GaussianNB()
grid_search = GridSearchCV(estimator=nb, param_grid=params, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Mejores parámetros encontrados
best_params = grid_search.best_params_
print("Mejores parámetros encontrados:", best_params)

# Entrenar modelo Naive Bayes con los mejores parámetros
best_nb = GaussianNB(**best_params)
best_nb.fit(X_train, y_train)

# Predicciones en el conjunto de prueba
y_pred = best_nb.predict(X_test)

# Métricas de evaluación
print("Precisión:", accuracy_score(y_test, y_pred))
print("Reporte de clasificación:\n", classification_report(y_test, y_pred))
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))

# Matriz de confusión mejorada
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Severo', 'Severo'], yticklabels=['No Severo', 'Severo'])
plt.xlabel('Predicción')
plt.ylabel('Realidad')
plt.title('Matriz de Confusión')
plt.show()

# Curva ROC y Área bajo la Curva (AUC)
y_probs = best_nb.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='orange', lw=2, label=f'AUC = {roc_auc_score(y_test, y_probs):.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.show()

# Análisis de Características Importantes (si aplicable)
if hasattr(best_nb, 'feature_importances_'):
    feature_importance = best_nb.feature_importances_
    features = X.columns
    df_importance = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
    df_importance = df_importance.sort_values(by='Importance', ascending=False)

    # Gráfico de barras para visualizar la importancia de las características
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=df_importance, palette='viridis')
    plt.title('Importancia de las Características')
    plt.show()

# Validación Cruzada y Métricas Adicionales
cv_scores = cross_val_score(best_nb, X, y, cv=5, scoring='accuracy')
print("Puntuaciones de Validación Cruzada:", cv_scores)
print("Precisión Promedio en Validación Cruzada:", np.mean(cv_scores))

# Histograma de probabilidades para clases
plt.figure(figsize=(10, 6))
plt.hist([best_nb.predict_proba(X_test)[:, 0], best_nb.predict_proba(X_test)[:, 1]], bins=20, color=['blue', 'orange'], alpha=0.7, label=['No Severo', 'Severo'])
plt.xlabel('Probabilidad')
plt.ylabel('Frecuencia')
plt.title('Histograma de Probabilidades para Clases')
plt.legend()
plt.show()
