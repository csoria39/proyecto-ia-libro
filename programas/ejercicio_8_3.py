# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import numpy as np
 
def pos_encoding(T, d):
    PE = np.zeros((T, d))
    for pos in range(T):
        for i in range(0, d, 2):
            ang = pos / (10000 ** (i/d))
            PE[pos, i]   = np.sin(ang)
            if i+1 < d: PE[pos, i+1] = np.cos(ang)
    return PE
 
def sim_coseno(u, v):
    return np.dot(u,v) / (np.linalg.norm(u)*np.linalg.norm(v)+1e-10)
 
PE = pos_encoding(T=64, d=32)
 
print('Similitudes coseno entre posición 0 y otras posiciones\n')
print(f"{'Distancia':>10}  {'Pos A':>6}  {'Pos B':>6}  {'Similitud':>12}  Barra")
print('-' * 55)
 
for pos_b in [1, 2, 4, 8, 16, 32]:
    sim = sim_coseno(PE[0], PE[pos_b])
    barra = '█' * int(max(0, sim) * 30)
    print(f'{pos_b:>10}  {0:>6}  {pos_b:>6}  {sim:>12.4f}  {barra}')
 
print()
print('Observación: la similitud decrece al aumentar la distancia.')
print('Esto permite al transformer aprender que tokens cercanos')
print('están más relacionados que tokens lejanos.')
