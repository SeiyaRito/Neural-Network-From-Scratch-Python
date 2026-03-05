import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from NeuralNetwork import NeuralNetwork

# Leer archivo CSV
data = pd.read_csv(r'C:\Users\seiya\OneDrive\Escritorio\mushrooms.csv')

# Codificar datos categóricos
data = pd.get_dummies(data)

# Dividir en entradas y salidas
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Dividir en conjunto de entrenamiento y prueba (70% - 30%) manualmente
np.random.seed(42)
indices = np.random.permutation(len(X))
train_size = int(0.7 * len(X))
training_idx, test_idx = indices[:train_size], indices[train_size:]
X_train, X_test = X[training_idx, :], X[test_idx, :]
y_train, y_test = y[training_idx], y[test_idx]

# Crear red neuronal con nueva configuración
epochs = 1500  # Número de épocas
learning_rate = 0.05  # Tasa de aprendizaje
architecture = [X_train.shape[1], 8, 4, 2, 1]  # Arquitectura de la red

nn = NeuralNetwork(architecture, learning_rate=learning_rate)

# Entrenar red neuronal con la nueva configuración
print("Iniciando el entrenamiento de la red neuronal...")
errors = nn.train(X_train, y_train, epochs=epochs)
print("Entrenamiento completado.")

# Visualización del error vs. época
plt.plot(errors)
plt.xlabel('Épocas de Entrenamiento')
plt.ylabel('Error de Entrenamiento')
plt.title('Progreso del Entrenamiento de la Red Neuronal')
plt.show()

# Evaluación con datos de prueba
test_error = 0
predicciones = []
for inputs, expected_output in zip(X_test, y_test):
    actual_output = nn.forward(inputs)
    test_error += nn.calculate_error(expected_output, actual_output)
    predicciones.append(actual_output)

print(f"El error total en los datos de prueba es: {test_error}")

# Visualización de las predicciones vs valores reales en los datos de prueba
plt.figure()
plt.plot(range(len(y_test)), y_test, 'b', label='Valores Reales')
plt.plot(range(len(y_test)), predicciones, 'r', label='Predicciones')
plt.xlabel('Índice de Muestra')
plt.ylabel('Valor (0: Comestible, 1: Venenoso)')
plt.title('Comparación de Predicciones y Valores Reales')
plt.legend()
plt.show()

# Calcular métricas adicionales manualmente
predicciones_binarias = [1 if p > 0.5 else 0 for p in predicciones]

# Exactitud
exactitud = sum(y_test == predicciones_binarias) / len(y_test)

# Precisión, Recall y F1-Score
TP = sum((y_test == 1) & (np.array(predicciones_binarias) == 1))
TN = sum((y_test == 0) & (np.array(predicciones_binarias) == 0))
FP = sum((y_test == 0) & (np.array(predicciones_binarias) == 1))
FN = sum((y_test == 1) & (np.array(predicciones_binarias) == 0))

precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"La exactitud con la que se clasificaron los hongos fue de: {exactitud}")
print(f"La precisión en la clasificación de los hongos fue de: {precision}")
print(f"El recall en la clasificación de los hongos fue de: {recall}")
print(f"El F1-Score en la clasificación de los hongos fue de: {f1}")
