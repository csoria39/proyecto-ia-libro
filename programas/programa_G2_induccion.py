# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
# =======================================================
# PROGRAMA G-2 — Cabezas de inducción en GPT-2 real
# Capítulo 16, sección 16.6
#
# Una secuencia de 50 tokens elegidos al azar no tiene ningún
# sentido: GPT-2 no puede predecirla... la primera vez. Si la
# repetimos, ¿aprovecha lo que acaba de ver? Medimos la pérdida
# en las dos copias y buscamos, entre sus 144 cabezas (12 bloques
# × 12), las que miran «al token que vino después de la vez
# anterior»: las cabezas de inducción (Olsson et al., 2022).
# Requisitos: programa_8bis_gpt2_real.py, la carpeta gpt2/ y
# matplotlib (para la figura).
# =======================================================
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fuente = open('programa_8bis_gpt2_real.py', encoding='utf-8').read()
exec(fuente.split('# EXPERIMENTOS')[0])     # pesos, tokenizador y forward()

N = 50
rng = np.random.RandomState(0)
R = list(rng.randint(1000, 30000, size=N))  # 50 tokens al azar
ids = [50256] + R + R                       # <|endoftext|>, copia 1, copia 2
print('Secuencia (principio):', decodificar(R[:12]).replace('\n', ' '), '...')

logits, (_, _, cache) = forward(np.array([ids]))
logp = np.log(softmax(logits[0, :-1].astype(np.float64)))
perdida = -logp[np.arange(len(ids) - 1), ids[1:]]      # pérdida de cada predicción
acierto = logp.argmax(-1) == np.array(ids[1:])
# La predicción en la posición i es del token i+1. Copia 1: tokens 2..50;
# copia 2: tokens 52..100 (el primero de cada copia no se puede deducir).
c1, c2 = slice(1, N), slice(N + 1, 2 * N)
print(f'\nPérdida media en la 1.ª copia: {perdida[c1].mean():.2f}   aciertos: {acierto[c1].mean():.0%}')
print(f'Pérdida media en la 2.ª copia: {perdida[c2].mean():.2f}   aciertos: {acierto[c2].mean():.0%}')
print(f'Referencia «al azar»: ln({V}) = {np.log(V):.2f}')

# Puntuaciones por cabeza. En la 2.ª copia, la posición i contiene el mismo
# token que la i − N; el que vino después de aquel está en la i − N + 1.
pos = np.arange(N + 2, 2 * N + 1)
induccion = np.array([[c['pesos'][0, h, pos, pos - N + 1].mean() for h in range(H)] for c in cache])
anterior = np.array([[c['pesos'][0, h, np.arange(1, 2 * N + 1), np.arange(0, 2 * N)].mean()
                      for h in range(H)] for c in cache])
print('\nCabezas de inducción (atención media al token que siguió a la vez anterior):')
for k, h in sorted(np.ndindex(N_BLQ, H), key=lambda kh: -induccion[kh])[:5]:
    print(f'   bloque {k + 1:2d}, cabeza {h + 1:2d}: {induccion[k, h]:.2f}')
print('Cabezas «token anterior» (atención media al token inmediatamente anterior):')
for k, h in sorted(np.ndindex(N_BLQ, H), key=lambda kh: -anterior[kh])[:3]:
    print(f'   bloque {k + 1:2d}, cabeza {h + 1:2d}: {anterior[k, h]:.2f}')
print(f'Media de las {N_BLQ * H} cabezas: inducción {induccion.mean():.2f}, '
      f'token anterior {anterior.mean():.2f}')

# Figura: pérdida por posición y mapa de la mejor cabeza de inducción
kb, hb = np.unravel_index(induccion.argmax(), induccion.shape)
# Letra grande y negra: la figura se imprime a 17 cm de ancho
plt.rcParams.update({'font.size': 17, 'axes.titlesize': 18, 'text.color': 'black', 'axes.labelcolor': 'black',
                     'xtick.color': 'black', 'ytick.color': 'black', 'axes.edgecolor': 'black'})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.8), dpi=150, layout='constrained')
a1.plot(np.arange(1, 2 * N + 1), perdida, color='#1A3A6B', lw=1.5)
a1.axvline(N + 0.5, color='#D97706', ls='--', lw=1.5)
a1.set_ylim(-0.5, perdida.max() * 1.15)
a1.text(N * 0.5, perdida.max() * 1.07, '1.ª copia', ha='center')
a1.text(N * 1.5, perdida.max() * 1.07, '2.ª copia', ha='center')
a1.set_xlabel('posición del token predicho'); a1.set_ylabel('pérdida (nats)')
a1.set_title('GPT-2 ante 50 tokens al azar, repetidos', fontweight='bold')
im = a2.imshow(cache[kb]['pesos'][0, hb], cmap='Blues', vmin=0, vmax=1)
a2.set_title(f'Atención del bloque {kb + 1}, cabeza {hb + 1}', fontweight='bold')
a2.set_xlabel('key: posición a la que se mira'); a2.set_ylabel('query: posición que pregunta')
fig.colorbar(im, ax=a2, shrink=0.8, label='peso de atención')
plt.savefig('figura_9q_induccion.png')
print(f'\nFigura guardada en figura_9q_induccion.png (bloque {kb + 1}, cabeza {hb + 1})')
