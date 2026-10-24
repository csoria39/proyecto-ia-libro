# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import numpy as np
 
# =======================================================
# NEURONA QUE APRENDE MEDIANTE GRADIENTE DESCENDENTE
# Problema: función lógica AND
# =======================================================
 
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
 
def derivada_sigmoid(z):
    s = sigmoid(z)
    return s * (1.0 - s)
 
# -- Datos de entrenamiento: tabla verdad del AND --------
X = np.array([[0, 0],   # 0 AND 0 = 0
              [0, 1],   # 0 AND 1 = 0
              [1, 0],   # 1 AND 0 = 0
              [1, 1]])  # 1 AND 1 = 1
y = np.array([0, 0, 0, 1], dtype=float)
 
# -- Inicialización de parámetros -----------------------
np.random.seed(0)
pesos = np.random.randn(2) * 0.3   # 2 pesos (uno por entrada)
sesgo = 0.0
lr    = 0.5                         # tasa de aprendizaje
 
print('=== ENTRENANDO NEURONA PARA APRENDER AND ===\n')
print(f'Pesos iniciales: {pesos.round(4)}  Sesgo: {sesgo:.4f}\n')
 
# -- Bucle de entrenamiento -----------------------------
for epoca in range(10_000):
    error_total = 0.0
    
    for xi, yi in zip(X, y):         # un ejemplo a la vez
        # === FORWARD PASS ===
        z = np.dot(xi, pesos) + sesgo
        pred = sigmoid(z)
        
        # === CÁLCULO DEL ERROR ===
        error = yi - pred            # cuánto nos hemos equivocado
        error_total += error**2
        
        # === BACKWARD PASS (gradientes) ===
        # Regla de la cadena aplicada a una neurona
        delta = error * derivada_sigmoid(z)
        
        # === ACTUALIZACIÓN DE PARÁMETROS ===
        pesos += lr * delta * xi    # actualizar cada peso
        sesgo += lr * delta         # actualizar sesgo
    
    # Mostrar progreso cada 1000 épocas
    if epoca % 1000 == 0:
        perdida_media = error_total / len(X)
        print(f'Época {epoca:5d} | Pérdida: {perdida_media:.6f}')
 
# -- Resultados -----------------------------------------
print(f'\nPesos finales: {pesos.round(4)}  Sesgo: {sesgo:.4f}')
print('\n=== PRUEBA FINAL ===')
print(f'{"Entradas":^20} {"Predicción":>12} {"Esperado":>10} {"Correcto":>10}')
print('-' * 55)
for xi, yi in zip(X, y):
    z    = np.dot(xi, pesos) + sesgo
    pred = sigmoid(z)
    ok   = '✓' if round(pred) == yi else '✗'
    print(f'{str(xi.tolist()):^20} {pred:12.4f} {int(yi):10d} {ok:>10}')
