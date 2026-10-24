# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import numpy as np
 
def relu(z):                     return np.maximum(0, z)
def leaky_relu(z, alpha=0.01):   return np.where(z > 0, z, alpha * z)
def elu(z, alpha=1.0):           return np.where(z > 0, z, alpha*(np.exp(z)-1))
 
valores = np.array([-3, -1, -0.5, 0, 0.5, 1, 3], dtype=float)
 
print(f"{'z':>6}  {'ReLU':>8}  {'LeakyReLU':>12}  {'ELU':>8}  Gradiente Leaky")
print('-' * 56)
for z in valores:
    r  = relu(z)
    lr = leaky_relu(z)
    e  = elu(z)
    # Derivada de Leaky ReLU: 1 si z>0, 0.01 si z<=0
    grad_lr = 1.0 if z > 0 else 0.01
    print(f'{z:>6.1f}  {r:>8.3f}  {lr:>12.4f}  {e:>8.4f}  {grad_lr:>14.2f}')
 
print()
print('ReLU: gradiente = 0 para z < 0 (neurona muerta)')
print('Leaky: gradiente = 0.01 para z < 0 (siempre aprende algo)')
