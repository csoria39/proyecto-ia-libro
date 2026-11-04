# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import numpy as np

# =======================================================
# FUNCIONES DE ACTIVACIÓN
# Cada función transforma un número de entrada en una salida
# =======================================================

def sigmoid(z):
    """Comprime cualquier número al rango (0, 1)"""
    return 1.0 / (1.0 + np.exp(-z))

def derivada_sigmoid(z):
    """Derivada de sigmoid — necesaria para el aprendizaje"""
    s = sigmoid(z)
    return s * (1.0 - s)

def relu(z):
    """Rectified Linear Unit — la más usada en capas ocultas"""
    return np.maximum(0.0, z)

def derivada_relu(z):
    """Derivada de ReLU: 1 donde z>0, 0 donde z≤0"""
    return (np.asarray(z) > 0).astype(float)   # asarray: funciona también con un número suelto

def tanh_act(z):
    """Tangente hiperbólica — rango (-1, 1)"""
    return np.tanh(z)

def softmax(z):
    """Convierte un vector en distribución de probabilidad"""
    # Restamos el máximo para evitar desbordamiento numérico
    e_z = np.exp(z - np.max(z))
    return e_z / e_z.sum()

# -- Demostración ---------------------------------------
print('=== FUNCIONES DE ACTIVACIÓN ===')
print(f'{"Entrada":>8}  {"Sigmoid":>9}  {"ReLU":>6}  {"Tanh":>8}')
print('-' * 40)
for z in [-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0]:
    print(f'{z:8.1f}  {sigmoid(z):9.4f}  {relu(z):6.2f}  {tanh_act(z):8.4f}')

# Softmax con un ejemplo
logits = np.array([2.0, 1.0, 0.5, -0.5])
probs  = softmax(logits)
print(f'\nSoftmax([2.0, 1.0, 0.5, -0.5]):')
print(f'  {probs.round(4)}')
print(f'  Suman: {probs.sum():.6f}')
