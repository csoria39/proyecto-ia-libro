# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import numpy as np
 
np.random.seed(42)
 
# =============================================================
# DATASET: el mismo que el Programa R-1 (seno con ruido)
# =============================================================
 
def generar_datos(n, ruido=0.25, semilla=None):
    if semilla:
        np.random.seed(semilla)
    X = np.sort(np.random.uniform(0, 1, n))
    y = np.sin(2 * np.pi * X) + np.random.normal(0, ruido, n)
    return X.reshape(-1, 1), y
 
 
X_tr, y_tr = generar_datos(20, semilla=42)      # 20 datos de entrenamiento
X_val, y_val = generar_datos(200, semilla=99)   # 200 datos de validacion
 
# =============================================================
# MLP para regresion con Dropout (inverted dropout) integrado
# Reutiliza el mismo optimizador Adam que MLPRegresion (Programa R-1)
# para que el modelo "Sin Dropout" sea EXACTAMENTE el mismo modelo
# propenso a overfitting que ya conocemos.
# =============================================================
 
def relu(z): return np.maximum(0, z)
def d_relu(z): return (z > 0).astype(float)
 
 
class MLPConDropout:
    def __init__(self, n_oculta, p_dropout=0.5, semilla=0):
        np.random.seed(semilla)
        escala = np.sqrt(2.0 / 1)
        self.W1 = np.random.randn(1, n_oculta) * escala
        self.b1 = np.zeros(n_oculta)
        self.W2 = np.random.randn(n_oculta, 1) * 0.1
        self.b2 = np.zeros(1)
        self.p = p_dropout  # probabilidad de APAGAR una neurona
 
        self.m = {'W1': np.zeros_like(self.W1), 'b1': np.zeros_like(self.b1),
                  'W2': np.zeros_like(self.W2), 'b2': np.zeros_like(self.b2)}
        self.v = {k: np.zeros_like(val) for k, val in self.m.items()}
        self.t = 0  # contador de pasos, para la correccion de sesgo de Adam
 
    def forward(self, X, entrenando=True):
        # aplica dropout SOLO si entrenando=True
        self.X = X
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = relu(self.Z1)
 
        if entrenando and self.p > 0:
            # Mascara binaria: True = neurona activa. Escalado 1/(1-p)
            # incluido en la propia mascara ("inverted dropout"): la
            # magnitud esperada de la activacion no cambia con o sin
            # dropout, asi que no hace falta tocar nada en inferencia.
            self.mask = (np.random.rand(*self.A1.shape) > self.p) / (1.0 - self.p)
            self.D1 = self.A1 * self.mask
        else:
            self.mask = None       # inferencia: todas las neuronas, sin mascara
            self.D1 = self.A1
 
        self.Z2 = self.D1 @ self.W2 + self.b2
        return self.Z2
 
    def backward(self, dZ2, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        # el gradiente tambien pasa por la mascara: el gradiente de una
        # neurona apagada es 0 (se multiplica por la misma mascara del forward)
        m = self.X.shape[0]
        dW2 = self.D1.T @ dZ2 / m
        db2 = dZ2.mean(axis=0)
        dD1 = dZ2 @ self.W2.T
        dA1 = dD1 * self.mask if self.mask is not None else dD1
        dZ1 = dA1 * d_relu(self.Z1)
        dW1 = self.X.T @ dZ1 / m
        db1 = dZ1.mean(axis=0)
 
        self.t += 1
        for nombre, grad in [('W1', dW1), ('b1', db1), ('W2', dW2), ('b2', db2)]:
            self.m[nombre] = beta1 * self.m[nombre] + (1 - beta1) * grad
            self.v[nombre] = beta2 * self.v[nombre] + (1 - beta2) * grad ** 2
            m_hat = self.m[nombre] / (1 - beta1 ** self.t)
            v_hat = self.v[nombre] / (1 - beta2 ** self.t)
            getattr(self, nombre)[...] -= lr * m_hat / (np.sqrt(v_hat) + eps)
 
    def entrenar_epoch(self, X, y, lr):
        pred = self.forward(X, entrenando=True)          # dropout activado
        err = pred - y.reshape(-1, 1)
        loss = float(np.mean(err ** 2))
        self.backward(2 * err / len(X), lr)
        return loss
 
    def evaluar(self, X, y):
        pred = self.forward(X, entrenando=False)          # dropout desactivado
        return float(np.mean((pred - y.reshape(-1, 1)) ** 2))
 
 
# =============================================================
# COMPARATIVA: mismo modelo (256 neuronas, propenso a overfitting
# de R-1), una vez sin Dropout y otras con Dropout p=0.3 y p=0.5
# =============================================================
 
EPOCAS = 20000
LR = 0.01
 
configs = [
    ('Sin Dropout (p=0.00)', 0.00),
    ('Dropout moderado (p=0.30)', 0.30),
    ('Dropout estandar (p=0.50)', 0.50),
]
 
print('=== EFECTO DEL DROPOUT (red de 256 neuronas) ===\n')
print(f"{'Configuracion':>28} {'Train':>10} {'Val':>10} {'Ratio':>8}  Diagnostico")
print('-' * 78)
 
for nombre, p in configs:
    red = MLPConDropout(256, p_dropout=p, semilla=42)
    for _ in range(EPOCAS):
        red.entrenar_epoch(X_tr, y_tr, LR)
    err_tr = red.evaluar(X_tr, y_tr)
    err_val = red.evaluar(X_val, y_val)
    ratio = err_val / (err_tr + 1e-10)
    if ratio > 2.5:
        diag = f'OVERFITTING (val={ratio:.1f}x train)'
    elif err_tr > 0.12:
        diag = 'underfitting'
    else:
        diag = 'buen ajuste'
    print(f'{nombre:>28} {err_tr:>10.5f} {err_val:>10.5f} {ratio:>7.2f}x  {diag}')
 
# =============================================================
# Verificacion del escalado invertido: la media de la activacion
# debe ser aproximadamente la misma con y sin dropout.
# =============================================================
 
print('\n--- Verificacion del escalado de Dropout ---')
np.random.seed(0)
A_prueba = np.ones((1000, 8))  # activaciones simuladas, todas a 1
 
p = 0.5
mascara = (np.random.rand(*A_prueba.shape) > p) / (1.0 - p)
salida_entrenamiento = A_prueba * mascara
salida_inferencia = A_prueba  # sin mascara ni escalado
 
print(f'Media activacion (entrenamiento, con mascara): {salida_entrenamiento.mean():.4f}')
print(f'Media activacion (inferencia, sin mascara):    {salida_inferencia.mean():.4f}')
print('Las medias son similares gracias al escalado 1/(1-p) ("inverted dropout").')
