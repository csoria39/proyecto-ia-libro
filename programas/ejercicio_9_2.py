# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
# =======================================================
# EJERCICIO 13.2 — El efecto de la temperatura, con el modelo real
# Carga los pesos que guardó el Programa 8 y genera desde la
# misma semilla con cinco temperaturas (top_p = 1.0 para aislar
# el efecto de la temperatura).
# =======================================================
import numpy as np

fuente = open('programa_8_nano_llm.py', encoding='utf-8').read()
exec(fuente.split('# -- Learning rate')[0])              # datos, parámetros, forward
exec('def greedy' + fuente.split('def greedy')[1]
                     .split("print('\\n=== TEXTO")[0])   # generar() y filtros
params.update(np.load('nano_llm_pesos.npz'))             # el modelo ya entrenado

# Cuánto se concentra la probabilidad del siguiente carácter
contexto = [C2I[c] for c in 'un modelo ']
logits = forward(np.array([contexto]))[0][0, -1]   # lote de 1; última posición
print('\nProbabilidad del carácter más probable tras «un modelo »:')
for tau in [0.2, 0.5, 0.8, 1.2, 2.0]:
    p = softmax(logits / tau)
    print(f'  τ = {tau:.1f}: {p.max():.3f}  («{I2C[int(p.argmax())]}»)')

print('\nTexto generado (semilla «un modelo», top_p = 1.0):')
for tau in [0.2, 0.5, 0.8, 1.2, 2.0]:
    np.random.seed(0)
    texto = generar('un modelo', n_chars=90, temperatura=tau, top_p=1.0)
    print(f'\n--- τ = {tau} ---')
    print(texto)
