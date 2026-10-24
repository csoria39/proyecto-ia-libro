# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
# EJERCICIO 16.3 — La cabeza «token anterior» en un texto normal
# El Programa G-2 encontró, con tokens al azar, una cabeza que mira casi
# siempre al token anterior. ¿Hace lo mismo con un texto de verdad?
import numpy as np

fuente = open('programa_8bis_gpt2_real.py', encoding='utf-8').read()
exec(fuente.split('# EXPERIMENTOS')[0])

texto = ('The neural network reads the sentence one token at a time. Each layer adds a little '
         'more information to the vectors, and the attention heads decide where to look.')
ids = codificar(texto)
_, (_, _, cache) = forward(np.array([ids]))
T = len(ids)
i = np.arange(1, T)
print(f'{T} tokens. Para cada cabeza: % de posiciones cuyo máximo de atención es el token anterior')
res = np.array([[np.mean(c['pesos'][0, h, i].argmax(-1) == i - 1) for h in range(H)] for c in cache])
for k, h in sorted(np.ndindex(N_BLQ, H), key=lambda kh: -res[kh])[:3]:
    print(f'   bloque {k + 1:2d}, cabeza {h + 1:2d}: {res[k, h]:.0%}')
print(f'   media de las {N_BLQ * H} cabezas: {res.mean():.0%}')
k, h = 4, 11                                   # bloque 5, cabeza 12 (numeradas desde 1)
print(f'\nBloque 5, cabeza 12, en las 8 primeras posiciones (a quién mira cada token):')
for t in range(1, 9):
    j = int(cache[k]['pesos'][0, h, t].argmax())
    peso = cache[k]['pesos'][0, h, t, j]
    print(f'   {decodificar([ids[t]])!r:12s} → {decodificar([ids[j]])!r:12s} ({peso:.2f})')
