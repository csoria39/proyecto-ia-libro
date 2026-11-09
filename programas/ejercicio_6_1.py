# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import numpy as np
from programa_3_mlp import CapaDensa, ReLU, Sigmoid, RedNeuronal   # Programa 3
 
np.random.seed(0)
 
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y_xor  = np.array([0, 1, 1, 0], dtype=float)  # XOR
y_xnor = np.array([1, 0, 0, 1], dtype=float)  # XNOR = NOT XOR
 
for nombre, y in [('XOR', y_xor), ('XNOR', y_xnor)]:
    np.random.seed(42)
    red = RedNeuronal([
        CapaDensa(2, 8), ReLU(),
        CapaDensa(8, 1), Sigmoid()
    ])
    hist = red.entrenar(X, y, epocas=5000, lr=0.1, verbose=False)
    pred = red.predecir(X).ravel()   # CORRECCION: aplanar para NumPy 2.x
    correcto = sum(round(float(p))==int(e) for p,e in zip(pred,y))
    print(f'{nombre}: pérdida={hist[-1]:.5f}  precision={correcto}/4  ', end='')
    preds_str = str([round(float(p)) for p in pred])
    print(f'preds={preds_str}')
 
print()
print('Observación: XNOR = NOT XOR.')
print('La misma arquitectura resuelve ambos porque ambos son')
print('no linealmente separables y tienen la misma complejidad.')
