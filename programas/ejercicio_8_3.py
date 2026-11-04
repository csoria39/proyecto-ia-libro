# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import numpy as np
 
def sim_coseno(u, v):
    nu = np.linalg.norm(u) + 1e-10
    nv = np.linalg.norm(v) + 1e-10
    return float(np.dot(u, v) / (nu * nv))
 
# Vectores de demostración (en 3D para visualizar)
v_ref    = np.array([1.0, 0.0, 0.0])         # referencia
v_parec  = np.array([0.9, 0.4, 0.0])          # muy similar a v_ref
v_perp   = np.array([0.0, 0.0, 1.0])          # perpendicular a los dos
v_opuest = np.array([-1.0, 0.0, 0.0])         # opuesto
v_igual  = np.array([3.0, 0.0, 0.0])          # misma dirección, distinta magnitud
 
vectores = [
    (v_ref, 'referencia [1,0,0]'),
    (v_parec, 'parecido [0.9,0.4,0]'),
    (v_perp, 'perpendicular [0,0,1]'),
    (v_opuest, 'opuesto [-1,0,0]'),
    (v_igual, 'misma dirección [3,0,0]'),
]
 
print('Similitudes coseno respecto a v_ref = [1, 0, 0]\n')
print(f"{'Vector':>32}  {'Sim. coseno':>12}  Interpretación")
print('-' * 65)
for v, nombre in vectores:
    sim = sim_coseno(v_ref, v)
    if sim > 0.9:   interp = 'muy similares'
    elif sim > 0.5: interp = 'bastante similares'
    elif sim > 0.1: interp = 'algo similares'
    elif sim > -0.1: interp = 'perpendiculares'
    else:            interp = 'opuestos'
    print(f'{nombre:>32}  {sim:>12.4f}  {interp}')
 
print(f'\nY entre los otros dos: sim(parecido, perpendicular) = {sim_coseno(v_parec, v_perp):.4f}')
print('\nObserva: la similitud coseno NO depende de la magnitud,')
print('solo de la dirección. [1,0,0] y [3,0,0] son identicos.')
