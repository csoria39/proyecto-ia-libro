# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import numpy as np
 
def softmax_temp(z, tau=1.0):
    z_t = z / tau
    e   = np.exp(z_t - z_t.max())
    return e / e.sum()
 
z = np.array([2.0, 1.5, 0.5])
print(f'Vector: {z}')
print(f"{'Tau':>8}  {'P(A)':>8}  {'P(B)':>8}  {'P(C)':>8}  Comportamiento")
print('-' * 58)
 
for tau, desc in [
    (0.1, 'muy determinista'),
    (0.5, 'algo concentrada'),
    (1.0, 'neutro (default)'),
    (2.0, 'algo uniforme'),
    (10.0,'casi uniforme'),
]:
    p = softmax_temp(z, tau)
    print(f'{tau:>8.1f}  {p[0]:>8.4f}  {p[1]:>8.4f}  {p[2]:>8.4f}  {desc}')
 
print()
print('Con tau=0.1: casi todo el peso en el mayor (2.0)')
print('Con tau=10:  los tres tokens tienen probabilidad similar')
print('En generacion de LLMs: tau=0.8 da buen balance')
