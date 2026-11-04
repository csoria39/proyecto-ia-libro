# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import numpy as np
 
def softmax(x): e=np.exp(x-x.max()); return e/e.sum()
 
def min_p_filtrar(probs, p_min):
    '''Conserva tokens con prob >= p_min * prob_maxima'''
    umbral = p_min * probs.max()
    mask = np.where(probs >= umbral, probs, 0)
    return mask / mask.sum()
 
def top_p_filtrar(probs, p):
    idx = np.argsort(probs)[::-1]
    acum = np.cumsum(probs[idx])
    n = int(np.searchsorted(acum, p)) + 1
    mask = np.zeros_like(probs)
    mask[idx[:n]] = probs[idx[:n]]
    return mask / mask.sum()
 
tokens = ['el', 'la', 'un', 'los', 'de', 'que', 'en', 'y', 'a', 'por']
logits = np.array([3.8, 3.2, 2.1, 1.8, 1.5, 1.0, 0.5, 0.2, -0.3, -0.8])
probs  = softmax(logits)
 
print('=== MIN-P vs TOP-P ===\n')
for p_min in [0.05, 0.1, 0.3]:
    filtrado = min_p_filtrar(probs, p_min)
    n_activos = (filtrado > 0).sum()
    print(f'Min-P={p_min}: {n_activos} tokens activos')
    for tok, p in zip(tokens, filtrado):
        if p > 0: print(f'    {tok}: {p:.3f}')
    print()
 
for p in [0.8, 0.9]:
    filtrado = top_p_filtrar(probs, p)
    n_activos = (filtrado > 0).sum()
    print(f'Top-P={p}: {n_activos} tokens activos')
