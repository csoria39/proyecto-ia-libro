# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
# =======================================================
# PROGRAMA G-4 — Ajuste fino de GPT-2 con TU backward
# Capítulo 16, sección 16.8
#
# En el Capítulo 15, el Nano-LLM entrenado con 83 frases no supo nada
# de las 10 frases apartadas: memorizaba. ¿Qué le faltaba? Aquí
# tomamos GPT-2, que ya «leyó» 40 GB de texto (preentrenamiento),
# y lo ajustamos con las MISMAS 83 frases usando la función
# forward_backward() del Programa 8, sin cambiar una línea. Luego
# medimos las 10 frases apartadas y comparamos.
# Requisitos: programa_8bis_gpt2_real.py, programa_8_nano_llm.py,
# la carpeta gpt2/ y unos 3 GB de memoria libre.
# =======================================================
import sys
import time
import numpy as np

PASOS = int(sys.argv[1]) if len(sys.argv) > 1 else 200

fuente = open('programa_8bis_gpt2_real.py', encoding='utf-8').read()
exec(fuente.split('# EXPERIMENTOS')[0])     # pesos de GPT-2, tokenizador y generar()

# -- El backward del Programa 8, tal cual -------------------------
# Del Programa 8 tomamos sus funciones (desde «Utilidades» hasta
# antes del bucle de entrenamiento). Funcionan con cualquier tamaño:
# usan las variables globales H, DH, D, N_BLQ, MASCARA_CAUSAL y
# params, que ahora son las de GPT-2.
p8 = open('programa_8_nano_llm.py', encoding='utf-8').read()
exec('# -- Utilidades' + p8.split('# -- Utilidades')[1].split('# -- Learning rate')[0])
# Los pesos leídos del archivo son de solo lectura y algunas matrices
# son trozos de otras (Q, K, V): para entrenar hacen falta copias propias.
params = {k: np.array(v, dtype=np.float32) for k, v in params.items()}
del gpt2          # los pesos originales ya no hacen falta: se libera memoria

# -- Los mismos datos que el Programa RZ-1 -------------------------
TEXTO = p8.split('TEXTO = """')[1].split('"""')[0]
FRASES = TEXTO.strip().split('\n')
VAL_IDX = list(range(4, len(FRASES), 9))
VAL = [FRASES[i] for i in VAL_IDX]
TRAIN = [f for i, f in enumerate(FRASES) if i not in VAL_IDX]
DATOS = np.array(codificar('\n' + '\n'.join(TRAIN) + '\n'))
print(f'Entrenamiento: {len(TRAIN)} frases = {len(DATOS)} tokens BPE | validación: {len(VAL)} frases')


def bits_por_caracter(frases):
    """Pérdida en bits por carácter, comparable con la del Nano-LLM: se
    predice cada frase y su salto de línea final a partir del salto inicial."""
    nats, chars = 0.0, 0
    for f in frases:
        ids = [50256] + codificar('\n' + f + '\n')
        lp = np.log(softmax(forward(np.array([ids]))[0][0, :-1].astype(np.float64)))
        nats -= lp[np.arange(1, len(ids) - 1), ids[2:]].sum()   # tras el '\n' inicial
        chars += len(f) + 1
    return nats / chars / np.log(2)


print(f'\nAntes del ajuste:   entrenamiento {bits_por_caracter(TRAIN):.2f} | '
      f'validación {bits_por_caracter(VAL):.2f} bits/carácter')
np.random.seed(0)
print('Muestra antes:', generar('la atención permite', n_tokens=25).replace('\n', ' / '))

# -- Ajuste fino: Adam, warm-up + coseno y recorte, como el Programa 8
BATCH, T, LR, CLIP = 4, 64, 1e-4, 1.0
m_adam = {k: np.zeros_like(v) for k, v in params.items()}
v_adam = {k: np.zeros_like(v) for k, v in params.items()}
np.random.seed(42)
t0 = time.time()
for paso in range(PASOS):
    lr = LR * min(1.0, (paso + 1) / 20) * 0.5 * (1 + np.cos(np.pi * paso / PASOS))
    inicios = np.random.randint(0, len(DATOS) - T - 1, size=BATCH)
    xb = np.stack([DATOS[i:i + T] for i in inicios])
    yb = np.stack([DATOS[i + 1:i + T + 1] for i in inicios])
    grads, loss = forward_backward(xb, yb)
    norma = np.sqrt(sum(float(np.dot(g.ravel(), g.ravel())) for g in grads.values()))
    if norma > CLIP:
        for k in grads:
            grads[k] *= CLIP / norma
    for k in params:
        m_adam[k] = 0.9 * m_adam[k] + 0.1 * grads[k]
        v_adam[k] = 0.999 * v_adam[k] + 0.001 * grads[k] ** 2
        params[k] -= lr * (m_adam[k] / (1 - 0.9 ** (paso + 1))) / (
            np.sqrt(v_adam[k] / (1 - 0.999 ** (paso + 1))) + 1e-8)
    if (paso + 1) % 25 == 0 or paso == 0:
        print(f'Paso {paso + 1:4d}: pérdida del lote = {loss:.3f} nats/token   norma = {norma:5.2f}   '
              f'validación = {bits_por_caracter(VAL):.2f} bits/carácter   ({time.time() - t0:.0f} s)',
              flush=True)

print(f'\nDespués del ajuste: entrenamiento {bits_por_caracter(TRAIN):.2f} | '
      f'validación {bits_por_caracter(VAL):.2f} bits/carácter')
print('Nano-LLM (Programa RZ-1), validación: mejor momento 1,923 nats = '
      f'{1.923 / np.log(2):.2f} bits; al final 4,175 nats = {4.175 / np.log(2):.2f} bits')
np.random.seed(0)
for semilla in ['la atención permite', 'la función softmax', 'un modelo']:
    print(f'\n--- {semilla!r} ---')
    print(generar(semilla, n_tokens=30))
