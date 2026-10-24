# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import numpy as np
 
def softmax(z):
    """Implementación estable de softmax"""
    e = np.exp(z - z.max())   # restar el máximo evita overflow
    return e / e.sum()
 
# Vectores de prueba, incluyendo casos extremos
vectores = [
    np.array([1.0, 2.0, 3.0]),
    np.array([100.0, 200.0, 300.0]),   # valores muy grandes
    np.array([-5.0, 0.0, 5.0]),
    np.array([0.0, 0.0, 0.0]),          # todos iguales -> uniforme
]
 
print('Verificación: softmax siempre suma 1.0')
print(f"{'Vector':>28}  {'Softmax':>26}  {'Suma':>10}")
print('-' * 68)
for v in vectores:
    s = softmax(v)
    print(f'{str(v):>28}  {str(s.round(4)):>26}  {s.sum():.8f}')
 
# Sin el truco del máximo, np.exp([100,200,300]) -> overflow
# Con el truco: exp([100-300, 200-300, 300-300]) = exp([-200,-100,0])
