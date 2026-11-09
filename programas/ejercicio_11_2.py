# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import numpy as np
 
def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x); return e / e.sum(axis=axis, keepdims=True)
 
def entropia(pesos):
    """Entropia media de las filas (bits). Mayor = más distribuida."""
    return -(pesos * np.log2(pesos + 1e-10)).sum(axis=1).mean()
 
np.random.seed(0)
T, d = 8, 64
X  = np.random.randn(T, d)
Wq = np.random.randn(d, d) * 0.02
Wk = np.random.randn(d, d) * 0.02
Q  = X @ Wq; K = X @ Wk
scores_raw    = Q @ K.T
scores_scaled = Q @ K.T / np.sqrt(d)
pesos_raw    = softmax(scores_raw)
pesos_scaled = softmax(scores_scaled)
 
print(f'Dimensión d={d}, T={T} tokens\n')
print(f"{'Métrica':>35}  {'Sin escala':>12}  {'Con escala':>12}")
print('-' * 62)
print(f"{'Varianza de scores':>35}  {scores_raw.var():>12.4f}  {scores_scaled.var():>12.4f}")
print(f"{'Max peso de atención':>35}  {pesos_raw.max():>12.4f}  {pesos_scaled.max():>12.4f}")
print(f"{'Entropía media (bits)':>35}  {entropia(pesos_raw):>12.4f}  {entropia(pesos_scaled):>12.4f}")
print(f"{'Máxima entropía posible':>35}  {np.log2(T):>12.4f}  {np.log2(T):>12.4f}")
print()
print('Sin escala: scores de alta varianza -> softmax muy puntuda')
print('-> gradientes casi nulos -> el modelo no aprende')
