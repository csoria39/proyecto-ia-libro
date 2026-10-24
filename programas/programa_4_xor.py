# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import numpy as np
from programa_3_mlp import CapaDensa, ReLU, Sigmoid, RedNeuronal   # Programa 3
 
# =======================================================
# RESOLVIENDO XOR CON UNA RED MULTICAPA
# =======================================================
 
np.random.seed(42)
 
# Datos del XOR
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([0,  1,  1,  0],          dtype=float)
 
# Arquitectura: 2 -> 8 -> 1
# La capa oculta de 8 neuronas da más que suficiente capacidad
red = RedNeuronal([
    CapaDensa(2, 8),   # capa oculta: 2 entradas -> 8 neuronas
    ReLU(),
    CapaDensa(8, 1),   # capa de salida: 8 -> 1
    Sigmoid()
])
 
print('=== RESOLVIENDO XOR CON MLP ===')
print(f'Arquitectura: 2 entradas -> 8 ocultas (ReLU) -> 1 salida (Sigmoid)')
print(f'Parámetros: {2*8+8 + 8*1+1} pesos + sesgos\n')
print('Entrenamiento:')
historico = red.entrenar(X, y, epocas=5000, lr=0.1)
 
print('\n=== RESULTADOS FINALES ===')
pred = red.predecir(X).ravel()   # CORRECCION: aplanar (4,1)->(4,) para NumPy 2.x
print(f'{"Entradas":^15} {"Predicción":>12} {"Redondeado":>12} {"Correcto":>10}')
print('-' * 53)
for xi, yi, pi in zip(X, y, pred):
    entradas_str = f'{int(xi[0])} XOR {int(xi[1])}'
    correcto = '✓' if round(float(pi)) == int(yi) else '✗'
    print(f'{entradas_str:^15} {float(pi):12.4f} {round(float(pi)):12d} {correcto:>10}')
 
print(f'\nPérdida final: {historico[-1]:.8f}')
