# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import numpy as np
 
# =======================================================
# CONVOLUCIÓN 2D MANUAL CON VISUALIZACIÓN ASCII
# Veremos cómo distintos filtros 'ven' la misma imagen
# =======================================================
 
def convolucion_2d(imagen, filtro, relleno=1):
    """
    Aplica un filtro a una imagen 2D.
    imagen:   (H, W)  — valores 0..1
    filtro:   (fH, fW)
    relleno:  píxeles de ceros añadidos al borde (para mantener tamaño)
    Retorna:  (H, W)  — mapa de características
    """
    H, W   = imagen.shape
    fH, fW = filtro.shape
    if relleno:
        imagen = np.pad(imagen, relleno, mode='constant', constant_values=0)
    H_r, W_r = imagen.shape
    H_out = H_r - fH + 1
    W_out = W_r - fW + 1
    resultado = np.zeros((H_out, W_out))
    for i in range(H_out):
        for j in range(W_out):
            region = imagen[i:i+fH, j:j+fW]
            resultado[i, j] = np.sum(region * filtro)
    return resultado
 
def max_pooling(feat_map, tamano=2):
    """
    Reduce el mapa de características tomando el máximo
    en cada ventana de tamanoxtamano.
    """
    H, W = feat_map.shape
    H_out = H // tamano
    W_out = W // tamano
    resultado = np.zeros((H_out, W_out))
    for i in range(H_out):
        for j in range(W_out):
            ventana = feat_map[i*tamano:(i+1)*tamano,
                               j*tamano:(j+1)*tamano]
            resultado[i, j] = ventana.max()
    return resultado
 
def relu(x):
    return np.maximum(0, x)
 
def a_ascii(matriz, umbral=0.3):
    """Convierte una matriz en arte ASCII para visualización"""
    filas = []
    vmax = np.abs(matriz).max() + 1e-8
    for fila in matriz:
        chars = []
        for v in fila:
            v_norm = v / vmax
            if v_norm > 0.5:  chars.append('█')
            elif v_norm > 0.2: chars.append('▓')
            elif v_norm > 0.0: chars.append('░')
            elif v_norm < -0.5: chars.append('v')
            elif v_norm < -0.2: chars.append('▽')
            else:              chars.append('.')
        filas.append('  ' + ' '.join(chars))
    return chr(10).join(filas)   # CORREGIDO: salto de linea, no espacio
 
 
# -- Definir una imagen de 12x12: una 'L' mayúscula ----
# (1 = blanco/activado, 0 = negro/apagado)
imagen = np.array([
  [0,0,0,0,0,0,0,0,0,0,0,0],
  [0,1,1,0,0,0,0,0,0,0,0,0],
  [0,1,1,0,0,0,0,0,0,0,0,0],
  [0,1,1,0,0,0,0,0,0,0,0,0],
  [0,1,1,0,0,0,0,0,0,0,0,0],
  [0,1,1,0,0,0,0,0,0,0,0,0],
  [0,1,1,0,0,0,0,0,0,0,0,0],
  [0,1,1,0,0,0,0,0,0,0,0,0],
  [0,1,1,1,1,1,1,1,0,0,0,0],
  [0,1,1,1,1,1,1,1,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0],
], dtype=float)
 
# -- Cuatro filtros clásicos de 3x3 --------------------
filtros = {
    'Bordes verticales (Sobel-X)': np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]], dtype=float),
 
    'Bordes horizontales (Sobel-Y)': np.array([
        [-1,-2,-1],
        [ 0, 0, 0],
        [ 1, 2, 1]], dtype=float),
 
    'Detector de esquinas (Laplaciano)': np.array([
        [ 0,-1, 0],
        [-1, 4,-1],
        [ 0,-1, 0]], dtype=float),
 
    'Suavizado (Gaussiano 3x3)': np.array([
        [1/16, 2/16, 1/16],
        [2/16, 4/16, 2/16],
        [1/16, 2/16, 1/16]], dtype=float),
}
 
print('=== CONVOLUCIÓN 2D: DISTINTOS FILTROS, MISMA IMAGEN ===\n')
print('Imagen original (12x12 — letra L):',)
print(a_ascii(imagen))
print()
 
for nombre, filtro in filtros.items():
    mapa = convolucion_2d(imagen, filtro, relleno=1)
    mapa_relu = relu(mapa)                   # solo activaciones positivas
    mapa_pool = max_pooling(mapa_relu, 2)    # reducir a la mitad
    print(f'Filtro: {nombre}')
    print(f'  Mapa de características (12x12, tras ReLU):')
    print(a_ascii(mapa_relu))
    print(f'  Tras Max-Pooling 2x2 -> (6x6):')
    print(a_ascii(mapa_pool))
    print()
