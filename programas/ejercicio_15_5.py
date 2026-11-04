# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
# =======================================================
# EJERCICIO 15.5 — ¿En qué se equivoca RZ-4 con 8 pasos?
# Entrena el Programa RZ-4 con la semilla 7 (la única de las
# seis de 15.5 que falla con 8 pasos) y clasifica cada error
# según el PRIMER token del razonamiento que difiere del correcto.
# Tiempo: el del Programa RZ-4 (~35 min) + 1 min de análisis.
# =======================================================
import sys
import numpy as np
from collections import Counter

sys.argv = [sys.argv[0], '7']          # RZ-4 lee la semilla de sus argumentos
exec(open('programa_RZ4_deduce_frases.py', encoding='utf-8').read())


def tipo_error(sal, correcto):
    for i, (a, b) in enumerate(zip(sal, correcto)):
        if a != b:
            break
    else:
        return 'razonamiento demasiado largo o corto'
    if i == len(correcto) - 1:
        return 'cadena correcta, respuesta final mal'
    if b in CONCEPTOS:
        return 'se equivoca de concepto (búsqueda del hecho)'
    if b in VERBOS:
        return 'se equivoca de verbo'
    if b in ('[+]', '[−]'):
        return 'se equivoca de signo (efecto acumulado)'
    return f'otro ({b!r} → {a!r})'


print('\n=== ANÁLISIS DE ERRORES CON 8 PASOS (200 problemas) ===')
rng = np.random.RandomState(1000 + 8)                  # los mismos del examen
probs = [problema(8, rng) for _ in range(200)]
sal = resolver([pr for pr, _ in probs])
errores = Counter()
posiciones = Counter()
ejemplo = None
for s, (pr, r) in zip(sal, probs):
    if s == r:
        continue
    t = tipo_error(s, r)
    errores[t] += 1
    i = next((k for k, (a, b) in enumerate(zip(s, r)) if a != b), len(r))
    posiciones[(i - 1) // 3 + 1] += 1                  # en qué paso de la cadena
    if ejemplo is None and 'concepto' in t:
        ejemplo = (pr, s, r)
print(f'Problemas con algún error: {sum(errores.values())} de 200')
for t, n in errores.most_common():
    print(f'  {n:3d}  {t}')
print('Paso de la cadena en el que aparece el primer error:',
      ', '.join(f'paso {k}: {v}' for k, v in sorted(posiciones.items())))
if ejemplo:
    pr, s, r = ejemplo
    i = next(k for k, (a, b) in enumerate(zip(s, r)) if a != b)
    print('\nEjemplo (primer error marcado con >>):')
    print('  correcto:', ' '.join(r[:i]), '>>', r[i], '...')
    print('  modelo:  ', ' '.join(s[:i]), '>>', s[i], '...')
