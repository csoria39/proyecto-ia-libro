# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import numpy as np
 
def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)
 
np.random.seed(7)
T, d = 6, 8
X  = np.random.randn(T, d)
Wq = np.random.randn(d, d) * 0.2
Wk = np.random.randn(d, d) * 0.2
 
Q      = X @ Wq
K      = X @ Wk
scores = Q @ K.T / np.sqrt(d)
pesos  = softmax(scores)
 
print('=== VERIFICACIÓN DE PESOS DE ATENCIÓN ===\n')
print(f'Matriz de pesos ({T}x{T}):')
np.set_printoptions(precision=4, suppress=True)
print(pesos)
 
print('\nVerificaciones:')
sumas = pesos.sum(axis=1)
print(f'  Todas las filas suman 1: {np.allclose(sumas, 1.0)}')
print(f'  Todos los pesos >= 0:    {(pesos >= 0).all()}')
print(f'  Suma mínima de fila:  {sumas.min():.10f}')
print(f'  Suma máxima de fila:  {sumas.max():.10f}')
print(f'  Peso mínimo: {pesos.min():.2e}  (positivo por softmax)')
