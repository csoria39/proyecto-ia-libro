# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import numpy as np
 
# =============================================================
# ESTRATEGIAS DE LEARNING RATE SCHEDULING
# Cada funcion recibe el numero de paso actual y
# devuelve el lr a usar en ese paso.
# =============================================================
 
def lr_constante(t, lr_max, **kw):
    return lr_max
 
def lr_step_decay(t, lr_max, step=100, factor=0.5, **kw):
    """Reduce el lr a la mitad cada 'step' epocas"""
    return lr_max * (factor ** (t // step))
 
def lr_exponencial(t, lr_max, decay=0.995, **kw):
    """Reduce el lr exponencialmente en cada paso"""
    return lr_max * (decay ** t)
 
def lr_coseno(t, lr_max, t_total, lr_min=1e-6, **kw):
    """
    Decaimiento en coseno: el lr baja suavemente siguiendo
    la mitad de un periodo del coseno.
    """
    return lr_min + 0.5*(lr_max - lr_min)*(1 + np.cos(np.pi*t/t_total))
 
def lr_warmup_coseno(t, lr_max, t_total, t_warmup=100, lr_min=1e-6, **kw):
    """
    Estandar en LLMs: sube linealmente durante t_warmup pasos,
    luego baja en coseno hasta lr_min.
    Motivo del warmup: al inicio los gradientes son inestables.
    Comenzar con lr alto puede causar divergencia.
    """
    if t < t_warmup:
        return lr_max * t / t_warmup          # subida lineal
    t_adj   = t - t_warmup
    t_decay = t_total - t_warmup
    return lr_min + 0.5*(lr_max - lr_min)*(1 + np.cos(np.pi*t_adj/t_decay))
 
 
# =============================================================
# VISUALIZACION DE LAS CURVAS DE LR
# =============================================================
T      = 500   # total de pasos
LR_MAX = 0.01
 
schedules = {
    'Constante':      lambda t: lr_constante(t, LR_MAX),
    'Step decay':     lambda t: lr_step_decay(t, LR_MAX, step=100),
    'Exponencial':    lambda t: lr_exponencial(t, LR_MAX),
    'Coseno':         lambda t: lr_coseno(t, LR_MAX, T),
    'Warmup+Coseno':  lambda t: lr_warmup_coseno(t, LR_MAX, T, t_warmup=50),
}
 
pasos_ref = [0, 25, 50, 100, 200, 300, 400, 500]
 
print('=== LEARNING RATE SCHEDULING ===')
print(f'lr_max={LR_MAX}, T={T} pasos\n')
print(f"{'Paso':>6}  ", end='')
for nombre in schedules: print(f'{nombre:>16}', end='')
print()
print('-' * (6 + 16*len(schedules) + 2))
 
for t in pasos_ref:
    print(f'{t:>6}  ', end='')
    for fn in schedules.values():
        print(f'{fn(t):>16.6f}', end='')
    print()
 
# Visualizacion ASCII de la curva Warmup+Coseno
fn_wc = schedules['Warmup+Coseno']
curva = [fn_wc(t) for t in range(T+1)]
print('\nWarmup+Coseno (el estandar en LLMs):')
print(f'  0.010 |', end='')
for t in range(0, T+1, 10):
    col = int(curva[t] / LR_MAX * 30)
    print('#' if col > 28 else ('|' if t==50 else ' '), end='')
print('|')
print(f'  0.005 |', end='')
for t in range(0, T+1, 10):
    col = int(curva[t] / LR_MAX * 30)
    marker = '#' if 14 < col <= 28 else ('.' if col==15 else ' ')
    print(marker, end='')
print('|')
print(f'  0.001 |' + '.' * 5 + '...warmup..|.......coseno decay......|')
print(f'         paso 0         50        100                       500')
 
print()
print('Durante el warmup (pasos 0-50): el lr sube de 0 a lr_max.')
print('Durante el decay (pasos 50-500): baja suavemente en coseno.')
print('Al final, lr ~ lr_min (casi cero) para ajuste fino preciso.')
