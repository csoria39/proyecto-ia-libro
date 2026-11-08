# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
"""Cifras del panel 5 del Diagrama 4.2: red 1-2-1 con sigmoide en las dos capas y pérdida L = ½(ŷ − y)².
Calcula forward y backward a mano (como el Capítulo 4) y comprueba los gradientes con diferencias finitas."""
import numpy as np

sig = lambda z: 1 / (1 + np.exp(-z))
x, y = 1.0, 0.8
W1 = np.array([[0.2], [-0.4]])      # 2 × 1
b1 = np.array([[0.0], [0.0]])
W2 = np.array([[0.7, -0.3]])        # 1 × 2
b2 = np.array([[0.1]])


def forward(W1, b1, W2, b2):
    z1 = W1 * x + b1
    a1 = sig(z1)
    z2 = W2 @ a1 + b2
    yh = sig(z2)
    return z1, a1, z2, yh, 0.5 * (yh - y) ** 2


z1, a1, z2, yh, L = forward(W1, b1, W2, b2)
d2 = (yh - y) * yh * (1 - yh)                 # δ⁽²⁾ = (ŷ − y)·f′(z⁽²⁾)
d1 = (W2.T @ d2) * a1 * (1 - a1)              # δ⁽¹⁾ = (W⁽²⁾)ᵀ δ⁽²⁾ ⊙ f′(z⁽¹⁾)
gW2, gb2, gW1, gb1 = d2 @ a1.T, d2, d1 * x, d1

print('Forward')
print('  z1 =', z1.ravel().round(4), '  a1 = f(z1) =', a1.ravel().round(4))
print(f'  z2 = {z2.item():.4f}   ŷ = f(z2) = {yh.item():.4f}   L = {L.item():.4f}')
print('Backward')
print(f'  δ2 = {d2.item():.4f}')
print('  δ1 =', d1.ravel().round(4))
print('  dL/dW2 =', gW2.ravel().round(4), f'  dL/db2 = {gb2.item():.4f}')
print('  dL/dW1 =', gW1.ravel().round(4), '  dL/db1 =', gb1.ravel().round(4))

# comprobación con diferencias finitas
eps, err = 1e-6, 0.0
P = [W1, b1, W2, b2]
G = [gW1, gb1, gW2, gb2]
for p, g in zip(P, G):
    for i in np.ndindex(p.shape):
        old = p[i]
        p[i] = old + eps; Lp = forward(*P)[4].item()
        p[i] = old - eps; Lm = forward(*P)[4].item()
        p[i] = old
        err = max(err, abs((Lp - Lm) / (2 * eps) - g[i]))
print(f'Diferencias finitas: error absoluto máximo {err:.1e}')
