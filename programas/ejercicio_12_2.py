# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import numpy as np
 
def layer_norm(X, eps=1e-5):
    """Normaliza cada fila a media 0, std 1."""
    mu    = X.mean(axis=-1, keepdims=True)   # media por fila
    sigma = X.std(axis=-1,  keepdims=True)   # std por fila
    return (X - mu) / (sigma + eps)
 
np.random.seed(5)
T, d = 5, 8
 
# Crear datos con distribución no estándar
X = np.random.randn(T, d) * 3 + 5   # media≈5, std≈3
 
print('ANTES de LayerNorm:')
print(f'  Media global:    {X.mean():.4f}  (≈5)')
print(f'  Std global:      {X.std():.4f}   (≈3)')
print(f'  Media por fila:  {X.mean(axis=1).round(3)}')
 
X_norm = layer_norm(X)
print()
print('DESPUÉS de LayerNorm:')
print(f'  Media global:    {X_norm.mean():.8f}  (≈0)')
print(f'  Std global:      {X_norm.std():.8f}   (≈1)')
print(f'  Media por fila:  {X_norm.mean(axis=1).round(6)}')
print(f'  Std por fila:    {X_norm.std(axis=1).round(6)}')
 
print()
print('LayerNorm garantiza activaciones bien escaladas,')
print('lo cual es crucial para entrenar redes muy profundas.')
