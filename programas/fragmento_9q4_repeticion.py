# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
# Fragmento 16.4 — Greedy con y sin penalización por repetición
# La penalización es la del Programa S-2 (14.5), copiada tal cual.
import numpy as np
fuente = open('programa_8bis_gpt2_real.py', encoding='utf-8').read()
exec(fuente.split('# EXPERIMENTOS')[0])


def greedy(semilla, n=40, rep_pen=1.0):
    tokens = codificar(semilla)
    generados = list(tokens)
    for _ in range(n):
        logits = forward(np.array([tokens]))[0][0, -1].astype(np.float64)
        if rep_pen != 1.0:                                  # como en el Programa S-2
            for t in set(generados[-20:]):
                if logits[t] > 0: logits[t] /= rep_pen
                else:             logits[t] *= rep_pen
        siguiente = int(np.argmax(logits))
        tokens.append(siguiente)
        generados.append(siguiente)
    return decodificar(tokens)


for pen in [1.0, 1.3]:
    print(f'--- repetition penalty = {pen} ---')
    print(greedy('I enjoy walking with my cute dog', rep_pen=pen).replace('\n', ' / '))
