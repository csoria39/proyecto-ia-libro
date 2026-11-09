# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import numpy as np

def sigmoid(z): return 1/(1+np.exp(-z))

# Pesos de la neurona AND que entrena el Programa 2 (Capítulo 4)
w1, w2, b = 7.28, 7.28, -11.01

print('Mapa de activación de la neurona AND')
print('(█ > 0.7,  ▓ entre 0.4 y 0.7,  ░ entre 0.2 y 0.4,  . < 0.2)')
print()

# Cuadrícula de -0.4 a 1.4 en pasos de 0.2 (incluye el 0 y el 1)
x_vals = np.arange(-0.4, 1.5, 0.2)
y_vals = np.arange(1.4, -0.5, -0.2)   # invertido para que (0,0) esté abajo

# Filas: x2 de arriba abajo; columnas: x1 de izquierda a derecha
print('x2 ^')
for x2 in y_vals:
    fila = f'{x2:>4.1f} |'
    for x1 in x_vals:
        z   = w1*x1 + w2*x2 + b
        sal = sigmoid(z)
        if sal > 0.7:   fila += '█'
        elif sal > 0.4: fila += '▓'
        elif sal > 0.2: fila += '░'
        else:           fila += '.'
    print(fila)
print('     +' + '-'*len(x_vals) + '-> x1')
print('      ' + ''.join(f'{v:>1.0f}' if abs(v-round(v))<0.01 else ' '
                          for v in x_vals))
print()
print('La frontera (▓) separa el punto (1,1) de los demás.')
print('Es una línea recta: la neurona es un clasificador lineal.')
