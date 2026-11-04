# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
# =======================================================
# EJERCICIO 13.7 — ¿Qué hace de verdad el recorte del gradiente?
# Ejecuta el Programa 8 registrando la norma del gradiente en
# cada paso, con recorte (CLIP = 1.0) o sin él.
# Uso:  python ejercicio_13_7.py            -> con recorte
#       python ejercicio_13_7.py sin_recorte -> sin recorte
# =======================================================
import sys
import numpy as np

SIN_RECORTE = len(sys.argv) > 1 and sys.argv[1] == 'sin_recorte'
fuente = open('programa_8_nano_llm.py', encoding='utf-8').read()
fuente = fuente.replace("    if norma > CLIP:", "    NORMAS.append(norma)\n    if norma > CLIP:")
if SIN_RECORTE:
    fuente = fuente.replace("CLIP   = 1.0 ", "CLIP   = float('inf')")
fuente = fuente.replace('nano_llm_pesos.npz', 'pesos_ejercicio_13_7.npz')
NORMAS = []
exec(fuente)
NORMAS = np.array(NORMAS)
print(f'\n{"SIN" if SIN_RECORTE else "CON"} recorte — norma del gradiente antes de recortar:')
print(f'  mediana {np.median(NORMAS):.2f} | máxima {NORMAS.max():.2f} | '
      f'pasos con norma > 1: {(NORMAS > 1).sum()} de {len(NORMAS)}')
for a, b in [(0, 1000), (1000, 6000), (6000, 12000)]:
    print(f'  pasos {a:5d}-{b - 1:5d}: mediana {np.median(NORMAS[a:b]):.2f}')
