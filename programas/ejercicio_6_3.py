# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import numpy as np
from programa_3_mlp import CapaDensa, ReLU, Sigmoid, RedNeuronal   # Programa 3
 
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([0, 1, 1, 0], dtype=float)
 
print('¿Cuántas neuronas ocultas necesita XOR?\n')
print(f"{'N ocultas':>12}  {'Éxitos/10':>10}  {'Pérdida media':>15}")
print('-' * 42)
 
for n_ocultas in [1, 2, 3, 4, 8]:
    exitos = 0
    perdidas = []
    for semilla in range(10):
        np.random.seed(semilla)
        red = RedNeuronal([
            CapaDensa(2, n_ocultas), ReLU(),
            CapaDensa(n_ocultas, 1), Sigmoid()
        ])
        hist = red.entrenar(X, y, epocas=3000, lr=0.2, verbose=False)
        pred = red.predecir(X).ravel()   # CORRECCION: aplanar para NumPy 2.x
        ok = all(round(float(p))==int(e) for p,e in zip(pred,y))
        if ok: exitos += 1
        perdidas.append(hist[-1])
    print(f'{n_ocultas:>12}  {exitos:>10}/10  {np.mean(perdidas):>15.5f}')
