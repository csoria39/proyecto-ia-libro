# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import numpy as np
 
def f(p):    return p[0]**2 + 10*p[1]**2
def grad(p): return np.array([2*p[0], 20*p[1]])
 
p = np.array([5.0, 5.0])
m = np.zeros(2); v = np.zeros(2)
beta1, beta2, eps, lr = 0.9, 0.999, 1e-8, 0.3
 
print('=== TRAYECTORIA DE ADAM ===')
print('f(x,y) = x^2 + 10y^2   (gradiente en y es 10x mayor que en x)')
print()
print(f"{'Paso':>5}  {'x':>8}  {'y':>8}  {'grad_x':>8}  {'grad_y':>8}  {'paso_x':>8}  {'paso_y':>8}")
print('-' * 65)
 
for t in range(1, 16):
    g = grad(p)
    m = beta1*m + (1-beta1)*g
    v = beta2*v + (1-beta2)*g**2
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    paso = lr * m_hat / (np.sqrt(v_hat) + eps)
    p_anterior = p.copy()
    p -= paso
    if t <= 10 or t == 15:
        print(f'{t:>5}  {p[0]:>8.4f}  {p[1]:>8.4f}  {g[0]:>8.3f}  {g[1]:>8.3f}  {paso[0]:>8.4f}  {paso[1]:>8.4f}')
 
print()
print('Observa: aunque grad_y es ~10x mayor que grad_x al principio,')
print('los PASOS en x e y son de magnitud similar.')
print('Adam normaliza por la magnitud historica de cada gradiente.')
