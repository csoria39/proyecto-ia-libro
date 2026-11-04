# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import numpy as np
 
np.random.seed(42)
 
# Dataset: mismo que el Programa R-1
def generar(n, ruido=0.25, s=None):
    if s: np.random.seed(s)
    X = np.sort(np.random.uniform(0, 1, n)).reshape(-1,1)
    y = np.sin(2*np.pi*X.ravel()) + np.random.normal(0, ruido, n)
    return X, y
 
X_tr, y_tr   = generar(20, s=42)
X_val, y_val = generar(200, s=99)
 
def relu(z):   return np.maximum(0, z)
def d_relu(z): return (z > 0).astype(float)
 
# =============================================================
# MLP CON REGULARIZACION L2 OPCIONAL
# El unico cambio respecto al MLP basico: añadimos
# 2 * lambda * W al gradiente en cada actualizacion.
# =============================================================
class MLPConL2:
    def __init__(self, n_oc, lam=0.0):
        np.random.seed(42)
        self.W1  = np.random.randn(1, n_oc) * np.sqrt(2)
        self.b1  = np.zeros(n_oc)
        self.W2  = np.random.randn(n_oc, 1) * 0.1
        self.b2  = np.zeros(1)
        self.m = {'W1': np.zeros_like(self.W1), 'b1': np.zeros_like(self.b1),
                  'W2': np.zeros_like(self.W2), 'b2': np.zeros_like(self.b2)}
        self.v = {k: np.zeros_like(val) for k, val in self.m.items()}
        self.t = 0
        self.lam = lam          # fuerza de la regularizacion L2
    
    def forward(self, X):
        self.X  = X
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = relu(self.Z1)
        return self.A1 @ self.W2 + self.b2
    
    def backward(self, dZ2, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        dW2 = self.A1.T @ dZ2 + 2*self.lam*self.W2  # <-- L2 (dZ2 ya trae el 1/N del MSE)
        db2 = dZ2.sum(axis=0)
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * d_relu(self.Z1)
        dW1 = self.X.T @ dZ1 + 2*self.lam*self.W1  # <-- L2
        db1 = dZ1.sum(axis=0)
        self.t += 1
        for nombre, grad in [('W1',dW1),('b1',db1),('W2',dW2),('b2',db2)]:
            self.m[nombre] = beta1*self.m[nombre] + (1-beta1)*grad
            self.v[nombre] = beta2*self.v[nombre] + (1-beta2)*grad**2
            m_hat = self.m[nombre] / (1 - beta1**self.t)
            v_hat = self.v[nombre] / (1 - beta2**self.t)
            getattr(self, nombre)[...] -= lr * m_hat / (np.sqrt(v_hat) + eps)
    
    def train(self, X, y, lr):
        pred  = self.forward(X)
        err   = pred - y.reshape(-1,1)
        loss  = float(np.mean(err**2))
        self.backward(2*err/len(X), lr)
        return loss
    
    def mse(self, X, y):
        return float(np.mean((self.forward(X)-y.reshape(-1,1))**2))
    
    def norma_pesos(self):
        return float(np.sum(self.W1**2) + np.sum(self.W2**2))
 
 
# Comparamos tres configuraciones del mismo modelo (256 neuronas)
configs = [
    ('Sin regularizacion  (lambda=0)',    0.0),
    ('L2 moderada         (lambda=0.001)',  0.001),
    ('L2 fuerte           (lambda=0.01)',   0.01),
]
 
print('=== EFECTO DE LA REGULARIZACION L2 ===')
print('Modelo: 256 neuronas ocultas (propenso a overfitting)\n')
print(f"{'Configuracion':>40}  {'Train':>8}  {'Val':>8}  {'|W|^2':>10}  Diagnostico")
print('-'*90)
 
for nombre, lam in configs:
    red = MLPConL2(256, lam=lam)
    for _ in range(20000):
        red.train(X_tr, y_tr, lr=0.01)
    err_tr  = red.mse(X_tr, y_tr)
    err_val = red.mse(X_val, y_val)
    norma   = red.norma_pesos()
    ratio   = err_val / (err_tr + 1e-10)
    if ratio > 2.5:
        diag = f'OVERFITTING (val={ratio:.1f}x train)'
    elif err_tr > 0.12:
        diag = 'underfitting'
    else:
        diag = 'buen ajuste'
    print(f'{nombre:>40}  {err_tr:>8.5f}  {err_val:>8.5f}  {norma:>10.1f}  {diag}')
 
# Analisis de los pesos
print('\nAnálisis de los pesos tras el entrenamiento:')
for nombre, lam in configs:
    red = MLPConL2(256, lam=lam)
    for _ in range(20000):
        red.train(X_tr, y_tr, lr=0.01)
    pesos = np.concatenate([red.W1.ravel(), red.W2.ravel()])
    print(f'  {nombre:>37}: max|w|={np.abs(pesos).max():.3f}  '
          f'mean|w|={np.abs(pesos).mean():.4f}')
