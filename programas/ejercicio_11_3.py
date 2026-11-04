# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import numpy as np
 
def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    return np.exp(x) / np.exp(x).sum(axis=axis, keepdims=True)
 
np.random.seed(3)
T, d, n_heads = 6, 16, 4
d_head = d // n_heads   # 4
X = np.random.randn(T, d)
 
salidas_cabezas = []
for h in range(n_heads):
    Wq_h = np.random.randn(d, d_head) * 0.02
    Wk_h = np.random.randn(d, d_head) * 0.02
    Wv_h = np.random.randn(d, d_head) * 0.02
    Q = X @ Wq_h; K = X @ Wk_h; V = X @ Wv_h
    scores = Q @ K.T / np.sqrt(d_head)
    msk    = np.triu(np.full((T,T), -1e9), k=1)
    pesos  = softmax(scores + msk)
    sal_h  = pesos @ V              # (T, d_head)
    salidas_cabezas.append(sal_h)
 
# Concatenar cabezas: (T, n_heads*d_head) = (T, d)
concat = np.concatenate(salidas_cabezas, axis=-1)   # (6, 16)
# Proyección final
Wo     = np.random.randn(d, d) * 0.02
salida = concat @ Wo                                 # (6, 16)
 
print(f'Multi-Head Attention: {n_heads} cabezas x d_head={d_head}')
print(f'Entrada:  {X.shape} -> Salida: {salida.shape}')
print(f'(la forma se mantiene — condición del bloque transformer)')
params_atencion = n_heads * 3*d*d_head + d*d
print(f'Parámetros: {n_heads}x3x{d}x{d_head} + {d}x{d} = {params_atencion}')
