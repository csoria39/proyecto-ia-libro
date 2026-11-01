# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
# =======================================================
# EJERCICIO 13.3 — La perplejidad del Nano-LLM
# PPL = exp(pérdida media por carácter). Se calcula con el
# modelo entrenado sobre TODO el texto de entrenamiento.
# =======================================================
import numpy as np






fuente = open('programa_8_nano_llm.py', encoding='utf-8').read()
exec(fuente.split('# -- Learning rate')[0])
params.update(np.load('nano_llm_pesos.npz'))

# Pérdida media en ventanas consecutivas de CTX caracteres
perdidas = []
for i in range(0, len(DATOS) - CTX - 1, CTX):
    x, y = DATOS[i:i + CTX], DATOS[i + 1:i + CTX + 1]
    p = softmax(forward(np.array([x]))[0][0])        # lote de 1 ventana
    perdidas.extend(-np.log(p[np.arange(CTX), y] + 1e-10))
L = np.mean(perdidas)
print(f'\nPérdida media en el texto de entrenamiento: {L:.4f}')
print(f'Perplejidad: e^{L:.4f} = {np.exp(L):.2f}\n')

print(f"{'Pérdida':>8}  {'PPL':>6}  Interpretación")
print('-' * 58)
for perd in [np.log(V), 3.0, 2.0, 1.5, 1.0, 0.5, L]:
    ppl = np.exp(perd)
    if ppl >= V - 0.5: interp = 'Al azar: no ha aprendido nada'
    elif ppl > 15:     interp = 'Algo de estructura local'
    elif ppl > 4:      interp = 'Patrones de carácter bien aprendidos'
    elif ppl > 1.8:    interp = 'Buen modelo: captura el estilo'
    else:              interp = 'Excelente... o sobreajuste (ver Capítulo 15)'
    print(f'{perd:8.4f}  {ppl:6.2f}  {interp}')
