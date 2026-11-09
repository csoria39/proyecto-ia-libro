# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import numpy as np
 
def sigmoid(z): return 1/(1+np.exp(-z))
def d_sig(z): s=sigmoid(z); return s*(1-s)
 
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([0, 0, 0, 1], dtype=float)
 
np.random.seed(42)
w = np.random.randn(2)*0.3; b = 0.0
curva = []
 
for ep in range(5000):
    L = sum((y[i]-sigmoid(np.dot(X[i],w)+b))**2 for i in range(4))/4
    if ep % 100 == 0: curva.append((ep, L))
    for xi, yi in zip(X, y):
        z = np.dot(xi, w)+b; d = (yi-sigmoid(z))*d_sig(z)
        w += 0.5*d*xi; b += 0.5*d
 
print('Curva de aprendizaje AND (lr=0.5)\n')
print(f"{'Época':>8}  {'Pérdida':>12}  {'Mejora abs.':>13}  {'Mejora rel.':>13}")
print('-' * 52)
for i, (ep, L) in enumerate(curva[:15]):
    if i == 0:
        mejora_abs = mejora_rel = 0
    else:
        mejora_abs = curva[i-1][1] - L
        mejora_rel = mejora_abs / curva[i-1][1] * 100
    print(f'{ep:>8}  {L:>12.6f}  {mejora_abs:>13.6f}  {mejora_rel:>12.2f}%')
 
print()
print('Observación: la mejora absoluta decrece con el tiempo.')
print('Las primeras 100 épocas eliminan la mayor parte del error.')
print('El resto del entrenamiento ajusta finamente los pesos.')
