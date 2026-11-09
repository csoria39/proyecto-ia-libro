# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import numpy as np
 
# =============================================================
# FUNCION DE ROSENBROCK: el test clasico para optimizadores.
# Es un valle largo y curvo con un unico minimo en (1, 1).
# f(x,y) = (1-x)^2 + 100*(y-x^2)^2
# El gradiente es muy diferente en las dos direcciones,
# lo que hace que SGD basico zigzaguee terriblemente.
# =============================================================
 
def f(p):      return (1-p[0])**2 + 100*(p[1]-p[0]**2)**2
def grad_f(p):
    dx = -2*(1-p[0]) - 400*p[0]*(p[1]-p[0]**2)
    dy = 200*(p[1]-p[0]**2)
    return np.array([dx, dy])
 
P0     = np.array([-0.5, 0.8])   # punto de partida
N      = 3000                     # pasos
LR     = 0.001
BETA   = 0.9
 
# --- SGD puro ---
p_sgd  = P0.copy()
hist_sgd = [float(f(p_sgd))]
for _ in range(N):
    p_sgd -= LR * grad_f(p_sgd)
    hist_sgd.append(float(f(p_sgd)))
 
# --- SGD + Momentum ---
p_mom  = P0.copy()
v      = np.zeros(2)
hist_mom = [float(f(p_mom))]
for _ in range(N):
    g  = grad_f(p_mom)
    v  = BETA * v + g
    p_mom -= LR * v
    hist_mom.append(float(f(p_mom)))
 
# =============================================================
# MOSTRAR CONVERGENCIA
# =============================================================
print('=== SGD vs SGD+MOMENTUM — Funcion de Rosenbrock ===')
print(f'Minimo en (1.0, 1.0), f=0.  Partimos de {P0.tolist()}\n')
 
print(f"{'Paso':>6}  {'SGD':>12}  {'SGD+Momentum':>14}  Pérdida del Momentum")
print('-' * 48)
for paso in [0, 50, 200, 500, 1000, 2000, 3000]:
    sgd_v = hist_sgd[paso]
    mom_v = hist_mom[paso]
    if sgd_v > 0 and mom_v > 0:
        ventaja = f'{sgd_v/mom_v:.3g} veces menor' if mom_v < sgd_v else 'igual'
    else:
        ventaja = 'convergido'
    print(f'{paso:>6}  {sgd_v:>12.6f}  {mom_v:>14.6f}  {ventaja}')
 
print(f'\nPosicion final SGD:          {p_sgd.round(4)}')
print(f'Posicion final SGD+Momentum: {p_mom.round(4)}')
print(f'Posicion optima:             [1.0, 1.0]')
