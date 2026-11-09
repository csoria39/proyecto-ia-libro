# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
"""Cifras del panel «La convolución — cómo funciona» del Diagrama G.1. La imagen publicada tiene 6 filas y solo 5
columnas aunque se rotula 6×6; aquí se completa con una sexta columna de ceros. Se aplica el filtro Sobel-X con la
función convolucion_2d del Programa 9A (correlación: el filtro no se invierte), sin relleno (6×6 → 4×4) y con
relleno=1, el valor por defecto del programa (6×6 → 6×6). Se ejecuta desde programas/."""
import numpy as np

import io, contextlib, sys, os
sys.path.insert(0, os.getcwd())
with contextlib.redirect_stdout(io.StringIO()):                # 9A ejecuta su demo al importarse
    from programa_9a_convolucion import convolucion_2d

imagen = np.array([[0, 0, 1, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0],
                   [0, 1, 1, 0, 0, 0],
                   [0, 1, 1, 1, 0, 0],
                   [0, 0, 0, 0, 0, 0]], dtype=float)
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])
H, W = imagen.shape
f = sobel_x.shape[0]
mapa = convolucion_2d(imagen, sobel_x, relleno=0)
print(f'Imagen {H}×{W}, filtro {f}×{f}, sin relleno → mapa {mapa.shape[0]}×{mapa.shape[1]}:')
print(mapa.astype(int))
m1 = convolucion_2d(imagen, sobel_x, relleno=1)
print(f'Con relleno=1 (Programa 9A) → mapa {m1.shape[0]}×{m1.shape[1]}:')
print(m1.astype(int))
v = imagen[0:3, 0:3]
terminos = ' + '.join(f'{int(v[a, b])}×({sobel_x[a, b]:+d})' for a in range(3) for b in range(3))
print(f'Posición (0,0), ventana de las filas 0-2 y columnas 0-2: {terminos} = {int((v * sobel_x).sum())}')
