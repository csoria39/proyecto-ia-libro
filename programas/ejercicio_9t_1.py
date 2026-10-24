# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
# =======================================================
# EJERCICIO 15.1 — Early Stopping de verdad en el Programa RZ-1
# Guarda los pesos del momento con MEJOR pérdida de validación
# y compara ese modelo con el del final del entrenamiento.
# =======================================================
import numpy as np

codigo = open("programa_RZ1_memoriza.py", encoding="utf-8").read()
# 1) Cada vez que mejora la validación, se copian los pesos
codigo = codigo.replace(
    "            mejor = (lv, paso)",
    "            mejor = (lv, paso)\n"
    "            mejores_pesos = {k: x.copy() for k, x in p.items()}")
# 2) Se ejecuta solo la Prueba 1 (entrenamiento con validación)
exec(codigo.split('# PRUEBA 2')[0])
# 3) Funciones de generación y de copia literal de la Prueba 3
exec(codigo.split("p8 = dict(np.load('nano_llm_pesos.npz'))")[1]
           .split('rng = np.random.RandomState(2026)')[0])

print('\n=== MODELO FINAL frente a MODELO CON EARLY STOPPING ===')
for nombre, pesos in [('final (paso 11999)', p), (f'mejor validación (paso {mejor[1]})', mejores_pesos)]:
    p8 = pesos                                   # generar_lote() usa p8
    rng = np.random.RandomState(2026)
    textos = generar_lote('mañana la ', 150, 10, rng)
    copia = np.mean([fraccion_copiada(t) for t in textos])
    print(f'\n{nombre}: entrenamiento {perdida(pesos, XT, YT, WT):.3f} | '
          f'validación {perdida(pesos, XV, YV, WV):.3f} | copia literal {copia:.0%}')
    print(textos[0])
