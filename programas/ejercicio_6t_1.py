# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import numpy as np
 
def simular_gradiente(W_hh, T, H):
    grad = np.ones(H) / np.sqrt(H)
    normas = [float(np.linalg.norm(grad))]
    for t in range(T):
        d_tanh = 0.5 * np.ones(H)
        grad = (W_hh.T @ grad) * d_tanh
        normas.append(float(np.linalg.norm(grad)))
    return normas
 
np.random.seed(0)
T = 60
UMBRAL = 1e-6
 
print('=== HORIZONTE DE MEMORIA SEGUN TAMAÑO DEL ESTADO OCULTO ===\n')
print(f"{'H (estado oculto)':>18}  {'Horizonte (pasos)':>18}")
print('-' * 40)
 
for H in [8, 32, 128]:
    W_hh = np.random.randn(H, H) * 0.05   # misma escala para todos
    normas = simular_gradiente(W_hh, T, H)
    horizonte = next((i for i,v in enumerate(normas) if v < UMBRAL), T)
    print(f'{H:>18}  {horizonte:>18}')
 
print()
print('El horizonte de memoria es similar entre tamaños de H')
print('porque depende de la ESCALA de los pesos (eigenvalues de W_hh),')
print('no tanto del numero de neuronas del estado oculto.')
print('Esto es justamente lo que motivo el diseño de LSTM/GRU:')
print('controlar explicitamente esa escala con puertas aprendidas.')
