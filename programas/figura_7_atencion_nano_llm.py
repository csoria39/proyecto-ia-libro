# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
# Figura del Capítulo 11: pesos de atención del Nano-LLM ya entrenado (Programa 8)
# sobre una frase del corpus: 2 bloques × 4 cabezas.
# Requiere nano_llm_pesos.npz (lo guarda el Programa 8).
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fuente = open('programa_8_nano_llm.py', encoding='utf-8').read()
exec(fuente.split('# -- Learning rate')[0])          # datos, parámetros, forward
params.update(np.load('nano_llm_pesos.npz'))

frase = 'la atención permite ver todo'[:CTX]
tok = np.array([[C2I[c] for c in frase]])           # lote de una frase
T = tok.shape[1]
_, (_, _, cache) = forward(tok)
P = [cache[k]['pesos'][0] for k in range(N_BLQ)]      # cada uno: (H, T, T)

etiq = [c if c != ' ' else '␣' for c in frase]
fig, axes = plt.subplots(N_BLQ, H, figsize=(16, 8.4), dpi=110, layout='constrained')
for k in range(N_BLQ):
    for c in range(H):
        ax = axes[k, c]
        im = ax.imshow(P[k][c], cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks(range(T)); ax.set_xticklabels(etiq, fontsize=7)
        ax.set_yticks(range(T)); ax.set_yticklabels(etiq, fontsize=7)
        ax.set_title(f'Bloque {k + 1} · cabeza {c + 1}', fontsize=12, color='#1A3A6B', fontweight='bold')
        if k == N_BLQ - 1:
            ax.set_xlabel('key: carácter al que se mira', fontsize=10)
        if c == 0:
            ax.set_ylabel('query: carácter que pregunta', fontsize=10)
fig.suptitle('Pesos de atención del Nano-LLM entrenado (Programa 8) — «' + frase + '»',
             fontsize=14, color='#1A3A6B', fontweight='bold')
fig.colorbar(im, ax=axes, shrink=0.6, label='peso de atención')
plt.savefig('figura_7_atencion.png', bbox_inches='tight')

# Resumen numérico para el texto
np.set_printoptions(precision=2, suppress=True)
print('frase:', repr(frase), 'T =', T)
io = frase.index('ó')
for k in range(N_BLQ):
    for c in range(H):
        dist = [i - int(P[k][c][i].argmax()) for i in range(T)]
        print(f'Bloque {k + 1}, cabeza {c + 1}: distancias al más atendido', dist)
        print(f'   peso máximo medio {P[k][c].max(1).mean():.2f} | máximo en el anterior: {dist.count(1)} de {T} '
              f'| en sí mismo: {dist.count(0)} | fila «ó»:', P[k][c][io, :io + 1])
