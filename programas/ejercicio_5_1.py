# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import numpy as np
 
def f(p):      return (1-p[0])**2 + 100*(p[1]-p[0]**2)**2
def grad_f(p):
    dx = -2*(1-p[0]) - 400*p[0]*(p[1]-p[0]**2)
    dy = 200*(p[1]-p[0]**2)
    return np.array([dx, dy])
 
P0, N, LR, BETA = np.array([-0.5, 0.8]), 2000, 0.001, 0.9
 
# --- Momentum estandar ---
p_mom = P0.copy(); v = np.zeros(2)
for _ in range(N):
    g = grad_f(p_mom)
    v = BETA*v + g
    p_mom -= LR*v
 
# --- Nesterov Momentum ---
p_nes = P0.copy(); v = np.zeros(2)
for _ in range(N):
    # Evaluar el gradiente en la posicion 'adelantada'
    p_adelantado = p_nes - LR * BETA * v
    g = grad_f(p_adelantado)
    v = BETA*v + g
    p_nes -= LR*v
 
print('=== MOMENTUM vs NESTEROV ===')
print(f'Momentum: f={f(p_mom):.6f}  pos={p_mom.round(4)}')
print(f'Nesterov: f={f(p_nes):.6f}  pos={p_nes.round(4)}')
print(f'Optimo:   f=0.0  pos=[1.0, 1.0]')
print()
print('Nesterov suele converger algo mas rapido porque "corrige"')
print('su trayectoria antes de comprometerse con el paso completo.')
