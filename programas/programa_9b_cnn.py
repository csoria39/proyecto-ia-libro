# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import numpy as np
 
# =======================================================
# CNN ENTRENADA DESDE CERO
# Tarea: ¿barra horizontal o barra vertical?
# Arquitectura: Conv(4 filtros 3x3) -> ReLU -> MaxPool(2x2)
#              -> Flatten -> Dense(1) -> Sigmoid
# =======================================================
 
np.random.seed(42)
 
# -- Generador de datos sintéticos ---------------------
def generar_dataset(n_por_clase=150):
    """
    Clase 0: imagen 10x10 con una barra HORIZONTAL de 2 px de alto
    Clase 1: imagen 10x10 con una barra VERTICAL de 2 px de ancho
    Añadimos ruido gaussiano para hacerlo más realista.
    """
    imagenes, etiquetas = [], []
    for _ in range(n_por_clase):
        # Clase 0: barra horizontal
        img = np.zeros((10, 10))
        fila = np.random.randint(2, 7)        # fila aleatoria (evita bordes)
        img[fila:fila+2, 1:9] = 1.0           # barra de 2 px de alto
        img += np.random.randn(10, 10) * 0.15 # ruido
        imagenes.append(np.clip(img, 0, 1))
        etiquetas.append(0)
        # Clase 1: barra vertical
        img = np.zeros((10, 10))
        col = np.random.randint(2, 7)          # columna aleatoria
        img[1:9, col:col+2] = 1.0              # barra de 2 px de ancho
        img += np.random.randn(10, 10) * 0.15
        imagenes.append(np.clip(img, 0, 1))
        etiquetas.append(1)
    idx = np.random.permutation(len(imagenes))
    X = np.array(imagenes)[idx]     # (N, 10, 10)
    y = np.array(etiquetas)[idx]    # (N,)
    return X, y
 
X, y = generar_dataset(150)
# División entrenamiento / validación
corte = int(0.8 * len(X))
X_tr, y_tr = X[:corte], y[:corte]
X_val, y_val = X[corte:], y[corte:]
print(f'Dataset: {len(X_tr)} entrenamiento, {len(X_val)} validación')
 
# -- Operaciones base ----------------------------------
def conv_forward(img, filtros, sesgos, relleno=1):
    """
    img:     (H, W)  — imagen de un canal
    filtros: (F, fH, fW) — F filtros
    sesgos:  (F,)
    retorna: (F, H, W)  — F mapas de características
    """
    H, W   = img.shape
    F, fH, fW = filtros.shape
    if relleno:
        img_p = np.pad(img, relleno)
    else:
        img_p = img
    H_r, W_r = img_p.shape
    H_o = H_r - fH + 1
    W_o = W_r - fW + 1
    out = np.zeros((F, H_o, W_o))
    for f in range(F):
        for i in range(H_o):
            for j in range(W_o):
                out[f,i,j] = np.sum(img_p[i:i+fH, j:j+fW] * filtros[f]) + sesgos[f]
    return out
 
def max_pool_forward(feat, tam=2):
    """feat: (F, H, W) -> (F, H//tam, W//tam)"""
    F, H, W = feat.shape
    H_o, W_o = H//tam, W//tam
    out  = np.zeros((F, H_o, W_o))
    mask = np.zeros_like(feat, dtype=bool)   # guardamos posición del máximo
    for f in range(F):
        for i in range(H_o):
            for j in range(W_o):
                ventana = feat[f, i*tam:(i+1)*tam, j*tam:(j+1)*tam]
                out[f,i,j] = ventana.max()
                # Máscara: ¿dónde está el máximo?
                pos = np.unravel_index(ventana.argmax(), ventana.shape)
                mask[f, i*tam+pos[0], j*tam+pos[1]] = True
    return out, mask
 
def max_pool_backward(dout, mask, tam=2):
    """Propaga el gradiente solo al píxel que fue el máximo"""
    F, H_o, W_o = dout.shape
    dfeat = np.zeros_like(mask, dtype=float)
    for f in range(F):
        for i in range(H_o):
            for j in range(W_o):
                dfeat[f, i*tam:(i+1)*tam, j*tam:(j+1)*tam] += (
                    mask[f, i*tam:(i+1)*tam, j*tam:(j+1)*tam] * dout[f,i,j]
                )
    return dfeat
 
def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-500,500)))
# -- Parámetros de la CNN ------------------------------
F  = 4     # número de filtros convolucionales
fS = 3     # tamaño del filtro: 3x3
 
# Inicialización He para los filtros
filtros = np.random.randn(F, fS, fS) * np.sqrt(2.0/(fS*fS))
sesgos  = np.zeros(F)
 
# Tras conv(relleno=1): (F, 10, 10)
# Tras maxpool(2):       (F, 5, 5)
# Tras flatten:          F*5*5 = 100 entradas
n_flat = F * 5 * 5
 
# Capas densas: 100 -> 16 -> 1
W1 = np.random.randn(n_flat, 16) * np.sqrt(2.0/n_flat)
b1 = np.zeros(16)
W2 = np.random.randn(16, 1) * np.sqrt(2.0/16)
b2 = np.zeros(1)
 
LR = 0.01
 
def forward_completo(img):
    """Pasa una imagen por toda la CNN"""
    # 1. Convolución + ReLU
    conv   = conv_forward(img, filtros, sesgos, relleno=1)
    relu_c = np.maximum(0, conv)              # (F, 10, 10)
    # 2. Max Pooling
    pool, mask = max_pool_forward(relu_c, tam=2)  # (F, 5, 5)
    # 3. Aplanar
    flat = pool.flatten()                     # (100,)
    # 4. Capa densa 1
    z1   = flat @ W1 + b1                     # (16,)
    a1   = np.maximum(0, z1)                  # ReLU
    # 5. Capa densa 2 -> salida
    z2   = a1 @ W2 + b2                       # (1,)
    pred = sigmoid(z2)                     # CORRECCION: array de 1 elemento (no escalar)
    cache = (img, conv, relu_c, mask, pool, flat, z1, a1, z2)
    return pred, cache
 
def backward_completo(pred, target, cache):
    global filtros, sesgos, W1, b1, W2, b2
    img, conv, relu_c, mask, pool, flat, z1, a1, z2 = cache
    # -- Pérdida BCE y su gradiente ---------------------
    # BCE calculada desde el logit (estable): log(1 + e^z) - target·z
    loss = np.logaddexp(0, z2) - target * z2
    # BCE + sigmoide: dL/dz2 = pred - target (la derivada de la sigmoide se cancela)
    # -- Capa 2 -----------------------------------------
    dz2  = pred - target                      # gradiente respecto al logit z2
    dW2  = a1[:, None] * dz2[None, :]
    db2  = dz2
    da1  = dz2 @ W2.T                         # (16,)
    # -- Capa 1 -----------------------------------------
    dz1  = da1 * (z1 > 0)                     # ReLU
    dW1  = flat[:, None] * dz1[None, :]
    db1  = dz1
    dflat = dz1 @ W1.T                        # (100,)
    # -- Unflatten -> Max Pool backward -----------------
    dpool = dflat.reshape(F, 5, 5)
    drelu_c = max_pool_backward(dpool, mask, tam=2)  # (F,10,10)
    # -- ReLU backward ----------------------------------
    dconv = drelu_c * (conv > 0)              # (F,10,10)
    # -- Convolución backward (gradiente de filtros) ----
    img_p = np.pad(img, 1)                    # con relleno
    dF = np.zeros_like(filtros)
    db_conv = np.zeros_like(sesgos)
    H_o, W_o = dconv.shape[1], dconv.shape[2]
    for f in range(F):
        db_conv[f] = dconv[f].sum()
        for i in range(H_o):
            for j in range(W_o):
                dF[f] += img_p[i:i+fS, j:j+fS] * dconv[f,i,j]
    # -- Actualizar parámetros --------------------------
    filtros -= LR * dF
    sesgos  -= LR * db_conv
    W1 -= LR * dW1; b1 -= LR * db1
    W2 -= LR * dW2; b2 -= LR * db2
    return float(loss[0])  # CORRECCION: extraer escalar del array de 1 elemento
 
# -- Bucle de entrenamiento -----------------------------
print('\nEntrenando CNN...')
print(f'Parámetros: filtros={filtros.size}+{sesgos.size} sesgos, dense={W1.size+b1.size+W2.size+b2.size}, '
      f'total={filtros.size+sesgos.size+W1.size+b1.size+W2.size+b2.size}')
print()
EPOCAS = 15
for epoca in range(EPOCAS):
    perm = np.random.permutation(len(X_tr))
    loss_total = 0.0
    for i in perm:
        pred, cache = forward_completo(X_tr[i])
        loss_total += backward_completo(pred, float(y_tr[i]), cache)
    # Validación
    correctos = sum(
        round(float(forward_completo(X_val[i])[0][0])) == y_val[i]
        for i in range(len(X_val)))
    acc_val = correctos / len(X_val)
    print(f'Época {epoca+1:2d}: pérdida={loss_total/len(X_tr):.4f}  '
          f'precisión validación={acc_val*100:.1f}%')
 
# -- Prueba final con imágenes concretas ----------------
print('\n=== PRUEBA VISUAL ===')
print('Imagen 1: barra horizontal')
img_h = np.zeros((10,10)); img_h[4:6, 1:9] = 1.0
pred_h, _ = forward_completo(img_h)
pred_h = float(pred_h[0])  # CORRECCION: extraer escalar
print(f'  Predicción: {pred_h:.4f}  '
      f'-> {"HORIZONTAL" if pred_h < 0.5 else "VERTICAL"} ✓' if pred_h < 0.5 else '✗')
 
print('Imagen 2: barra vertical')
img_v = np.zeros((10,10)); img_v[1:9, 4:6] = 1.0
pred_v, _ = forward_completo(img_v)
pred_v = float(pred_v[0])  # CORRECCION: extraer escalar
print(f'  Predicción: {pred_v:.4f}  '
      f'-> {"VERTICAL" if pred_v > 0.5 else "HORIZONTAL"} ✓' if pred_v > 0.5 else '✗')
 
# -- Visualizar filtros aprendidos ----------------------
print('\n=== FILTROS APRENDIDOS (3x3) ===')
print('(valores positivos █, neutros ., negativos ▽)')
for f in range(F):
    filt = filtros[f]
    vmax = np.abs(filt).max() + 1e-8
    print(f'  Filtro {f+1}:', end=' ')
    for fila in filt:
        row = ''.join('█' if v/vmax > 0.3 else '▽' if v/vmax < -0.3 else '.' for v in fila)
        print(row, end='  ')
    print()
