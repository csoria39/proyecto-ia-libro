# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
# =======================================================
# EJERCICIO 9.6 — Verifica tu backward con diferencias finitas
# Compara el gradiente analítico de forward_backward() con el
# gradiente numérico (L(w+ε) − L(w−ε)) / 2ε en elementos al azar
# de TODOS los tensores del Programa 8.
# =======================================================
import numpy as np

# Cargamos del Programa 8 solo la parte anterior al entrenamiento
# (datos, parámetros, utilidades, forward y forward_backward) y
# después los pesos que guardó al terminar (nano_llm_pesos.npz).
fuente = open('programa_8_nano_llm.py', encoding='utf-8').read()
exec(fuente.split('# -- Learning rate')[0])
# Con los pesos ya entrenados los gradientes son mayores que en la
# inicialización y la comparación numérica es más fiable.
params.update(np.load('nano_llm_pesos.npz'))

rng = np.random.RandomState(0)
inicios = rng.randint(0, len(DATOS) - CTX - 1, size=2)      # un lote de 2 ventanas
x = np.stack([DATOS[i:i + CTX] for i in inicios])
y = np.stack([DATOS[i + 1:i + CTX + 1] for i in inicios])


def comprobar(n_muestras=6, eps=1e-5):
    """Error relativo ||num − ana|| / (||num|| + ||ana||) por tensor."""
    grads, _ = forward_backward(x, y)
    errores = {}
    for nombre, P in params.items():
        num, ana = [], []
        for _ in range(n_muestras):
            j = tuple(rng.randint(s) for s in P.shape)
            orig = P[j]
            P[j] = orig + eps; l_mas = forward_backward(x, y)[1]
            P[j] = orig - eps; l_menos = forward_backward(x, y)[1]
            P[j] = orig
            num.append((l_mas - l_menos) / (2 * eps))
            ana.append(grads[nombre][j])
        num, ana = np.array(num), np.array(ana)
        if max(abs(num).max(), abs(ana).max()) < 1e-8:
            errores[nombre] = None          # los dos gradientes son ≈ 0
        else:
            errores[nombre] = np.linalg.norm(num - ana) / (np.linalg.norm(num) + np.linalg.norm(ana))
    return errores


def maximo(errores):
    return max(e for e in errores.values() if e is not None)


errores = comprobar()
print('\nTensor      error relativo')
for nombre, e in errores.items():
    print(f'{nombre:10s}  ' + ('gradiente ≈ 0 (numérico y analítico)' if e is None else f'{e:.1e}'))
print(f'\nError relativo máximo: {maximo(errores):.1e}')

# ¿Y si olvidamos la regla del producto en la derivada de GELU?
def gelu_bwd(x):                                     # versión incompleta
    t = np.tanh(C_GELU * (x + 0.044715 * x * x * x))
    return 0.5 * (1 + t)                             # falta el segundo término

rng = np.random.RandomState(0)
errores_mal = comprobar()
print('\nSin el segundo término de la derivada de GELU:')
for nombre in ['W10', 'b10', 'W20', 'Wq0', 'wte']:
    print(f'{nombre:10s}  {errores_mal[nombre]:.1e}')
print(f'Error relativo máximo: {maximo(errores_mal):.1e}')
