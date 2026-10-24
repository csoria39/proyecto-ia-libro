# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import numpy as np
fuente = open('programa_O2_optimizadores.py', encoding='utf-8').read()
exec(fuente.split("print('Optimizadores")[0])   # las clases del Programa O-2
 
np.random.seed(42)
 
# =============================================================
# PROBLEMA: clasificacion de dos lunas (no lineal)
# Generamos dos semilunares entrelazadas — clasico para
# demostrar la diferencia entre optimizadores.
# =============================================================
def dos_lunas(n=300, ruido=0.15):
    """Genera dos semilunares entrelazadas"""
    n_half = n // 2
    # Luna superior
    ang1 = np.linspace(0, np.pi, n_half)
    X1   = np.column_stack([np.cos(ang1), np.sin(ang1)])
    X1  += np.random.normal(0, ruido, X1.shape)
    # Luna inferior (desplazada)
    ang2 = np.linspace(0, np.pi, n_half)
    X2   = np.column_stack([1 - np.cos(ang2), 0.5 - np.sin(ang2)])
    X2  += np.random.normal(0, ruido, X2.shape)
    X = np.vstack([X1, X2])
    y = np.array([0]*n_half + [1]*n_half)
    idx = np.random.permutation(n)
    return X[idx], y[idx]
 
X, y = dos_lunas(300)
X_tr, y_tr   = X[:240], y[:240]
X_val, y_val = X[240:], y[240:]
 
# =============================================================
# RED NEURONAL COMUN PARA TODOS LOS OPTIMIZADORES
# =============================================================
def sigmoid(z): return 1/(1+np.exp(-np.clip(z,-500,500)))
def relu(z):    return np.maximum(0, z)
def d_relu(z):  return (z > 0).astype(float)
 
def inicializar(s=0):
    np.random.seed(s)
    return {
        'W1': np.random.randn(2, 16) * np.sqrt(2/2),
        'b1': np.zeros(16),
        'W2': np.random.randn(16, 1) * 0.1,
        'b2': np.zeros(1),
    }
 
def forward(X, p):
    Z1 = X @ p['W1'] + p['b1']
    A1 = relu(Z1)
    Z2 = A1 @ p['W2'] + p['b2']
    return sigmoid(Z2), Z1, A1
 
def backward(X, y, pred, Z1, A1, p):
    m    = len(X)
    dZ2  = (pred - y.reshape(-1,1)) / m
    grads = {
        'W2': A1.T @ dZ2,
        'b2': dZ2.sum(axis=0),
    }
    dA1  = dZ2 @ p['W2'].T
    dZ1  = dA1 * d_relu(Z1)
    grads['W1'] = X.T @ dZ1
    grads['b1'] = dZ1.sum(axis=0)
    return grads
 
def bce_loss(pred, y):
    p = np.clip(pred.ravel(), 1e-7, 1-1e-7)
    return -np.mean(y*np.log(p) + (1-y)*np.log(1-p))
 
def accuracy(X, y, p):
    pred, _, _ = forward(X, p)
    return np.mean((pred.ravel()>0.5) == y)
 
# =============================================================
# COMPARATIVA: mismos datos, misma red, distintos optimizadores
# =============================================================
configs = [
    ('SGD              lr=0.1 ', SGD(lr=0.1)),
    ('SGD+Momentum     lr=0.1 ', SGDMomentum(lr=0.1, beta=0.9)),
    ('RMSprop          lr=0.01', RMSprop(lr=0.01)),
    ('Adam             lr=0.01', Adam(lr=0.01)),
    ('AdamW            lr=0.01', AdamW(lr=0.01, wd=0.01)),
]
 
EPOCAS   = 500
BATCH    = 32
 
print('=== COMPARATIVA DE OPTIMIZADORES ===')
print(f'Problema: clasificacion de dos lunas (no lineal)')
print(f'Red: 2 -> 16 (ReLU) -> 1 (Sigmoid)  |  {EPOCAS} epocas\n')
print(f"{'Optimizador':>30}  {'Loss ep.10':>12}  {'Loss ep.100':>12}  ",
      f"{'Loss ep.500':>12}  {'Acc val':>8}")
print('-' * 85)
 
for nombre, opt in configs:
    params = inicializar(s=42)
    hist   = []
    for ep in range(EPOCAS):
        # Mini-batch SGD
        idx = np.random.permutation(len(X_tr))
        for start in range(0, len(X_tr), BATCH):
            xb = X_tr[idx[start:start+BATCH]]
            yb = y_tr[idx[start:start+BATCH]]
            pred, Z1, A1 = forward(xb, params)
            grads = backward(xb, yb, pred, Z1, A1, params)
            opt.paso(params, grads)
        pred_tr, _, _ = forward(X_tr, params)
        hist.append(bce_loss(pred_tr, y_tr))
    
    acc = accuracy(X_val, y_val, params)
    l10  = hist[9]  if len(hist)>9  else hist[-1]
    l100 = hist[99] if len(hist)>99 else hist[-1]
    l500 = hist[-1]
    print(f'{nombre:>30}  {l10:>12.5f}  {l100:>12.5f}  {l500:>12.5f}  {acc:>8.1%}')
