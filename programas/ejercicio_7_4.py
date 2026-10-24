# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import numpy as np
 
def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()
 
tokens = ['el', 'gato', 'duerme']
K = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
V = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
q_gato = np.array([0.0, 2.0])          # "gato" busca su determinante
 
np.set_printoptions(precision=3, suppress=True)
scores = K @ q_gato / np.sqrt(2)
print('scores           :', scores)
print('pesos            :', softmax(scores))
print('salida           :', softmax(scores) @ V)
# Con máscara causal, "gato" (posición 2) no puede ver "duerme" (posición 3)
pesos_c = softmax(scores + np.array([0, 0, -1e9]))
print('pesos (causal)   :', pesos_c)
print('salida (causal)  :', pesos_c @ V)
