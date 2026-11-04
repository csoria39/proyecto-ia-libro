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
 
sims = [sim_coseno(PE[0], PE[k]) for k in range(1, 64)]
subidas = [k + 1 for k in range(len(sims) - 1) if sims[k + 1] > sims[k]]
print()
print(f'De la distancia 1 a la 4, la similitud cae de {sims[0]:.2f} a {sims[3]:.2f}.')
print(f'Pero no decrece siempre: entre las distancias 1 y 63 vuelve a subir')
print(f'{len(subidas)} veces (por ejemplo, de la 16 a la 32: {sims[15]:.2f} -> {sims[31]:.2f}).')
print('Las posiciones cercanas se parecen mas que la mayoria de las lejanas,')
print('pero es una tendencia, no una ley: el modelo aprende a usarla.')
