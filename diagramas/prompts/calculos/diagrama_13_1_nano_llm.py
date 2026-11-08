# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
"""Cifras del Diagrama 13.1 a partir del Programa 8 (semilla 42), sin cambiar el programa:
  1) vuelve a entrenarlo registrando la pérdida de cada paso (se añade una línea que la guarda) y da la media
     móvil centrada de 200 pasos en los pasos 0, 1000, …, 12 000, la que dibuja la curva de aprendizaje;
  2) con los pesos entrenados, las probabilidades del carácter siguiente a 'un modelo ' con τ = 0.5, 0.8 y 2.0,
     y cuántos caracteres conserva top-p = 0.9;
  3) la codificación de 'la red' y 'un modelo '.
Se ejecuta desde programas/ (lee programa_8_nano_llm.py). Tarda lo mismo que el Programa 8."""
import numpy as np, io, contextlib

fuente = open('programa_8_nano_llm.py', encoding='utf-8').read()
fuente = fuente.replace("    grads_acc, loss = forward_backward(xb, yb)\n",
                        "    grads_acc, loss = forward_backward(xb, yb)\n    PERDIDAS.append(loss)\n")
fuente = fuente.split('# -- Estrategias de muestreo')[0]       # sin generar texto ni guardar pesos
g = {'PERDIDAS': []}
with contextlib.redirect_stdout(io.StringIO()):
    exec(fuente, g)
P = np.array(g['PERDIDAS'])
np.savetxt('perdidas_programa_8.txt', P, fmt='%.4f')     # la pérdida de cada paso
print(f'Pasos: {len(P)}  pérdida paso 0 = {P[0]:.4f}  ln 46 = {np.log(46):.4f}  último paso = {P[-1]:.4f}')
print('Media móvil centrada de 200 pasos (paso 0: valor del propio paso):')
for paso in range(0, 12001, 1000):
    if paso == 0:
        v = P[0]
    else:
        a, b = max(0, paso - 100), min(len(P), paso + 100)
        v = P[a:b].mean()
    print(f'  paso {paso:6d}: {v:.2f}')

softmax, forward, C2I, I2C, params = g['softmax'], g['forward'], g['C2I'], g['I2C'], g['params']
ctx = [C2I[c] for c in 'un modelo ']
logits = forward(np.array([ctx]))[0][0, -1]
for tau in (0.5, 0.8, 2.0):
    p = softmax(logits / tau)
    top = np.argsort(p)[::-1][:4]
    print(f"τ = {tau}: " + '  '.join(f"P('{I2C[i]}')={p[i]:.2f}" for i in top))
p = softmax(logits / 0.8)
n = int(np.searchsorted(np.cumsum(np.sort(p)[::-1]), 0.9)) + 1
print(f'top-p 0.9 con τ = 0.8 conserva {n} caracteres')
print("'la red' ->", [C2I[c] for c in 'la red'])
print("'un modelo ' ->", ctx)
