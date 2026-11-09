# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import numpy as np
 
def convolucion_2d(img, filtro, relleno=1):
    H, W = img.shape; fH, fW = filtro.shape
    if relleno: img = np.pad(img, relleno)
    H_r, W_r = img.shape
    out = np.zeros((H_r-fH+1, W_r-fW+1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i,j] = np.sum(img[i:i+fH, j:j+fW] * filtro)
    return out
 
# Filtro diagonal principal \
filtro_diag = np.array([
    [ 2, -1, -1],
    [-1,  2, -1],
    [-1, -1,  2]], dtype=float)
 
# Imagen 1: diagonal (np.eye)
img_diag = np.eye(10)
# Imagen 2: barra horizontal
img_hor  = np.zeros((10,10)); img_hor[4:6, :] = 1.0
 
for nombre, img in [('diagonal', img_diag), ('barra horiz.', img_hor)]:
    resultado = np.maximum(0, convolucion_2d(img, filtro_diag, 1))
    activacion_total = resultado.sum()
    print(f'Imagen {nombre}: activación total = {activacion_total:.2f}')
 
print()
print('El filtro debe activarse mucho más sobre la diagonal')
print('que sobre la barra horizontal.')
print()
print('Variante: filtro diagonal secundaria /')
filtro_anti = np.array([[-1,-1,2],[-1,2,-1],[2,-1,-1]], dtype=float)
print('  filtro_anti =', filtro_anti.tolist())
