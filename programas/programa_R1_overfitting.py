# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import numpy as np
 
np.random.seed(42)
 
# =============================================================
# DATASET: funcion seno con ruido gaussiano
# Usamos POCOS datos de entrenamiento a proposito,
# para que el overfitting emerja facilmente.
# =============================================================
 
def generar_datos(n, ruido=0.25, semilla=None):
    if semilla: np.random.seed(semilla)
    X = np.sort(np.random.uniform(0, 1, n))
    y = np.sin(2 * np.pi * X) + np.random.normal(0, ruido, n)
    return X.reshape(-1, 1), y
 
X_tr,  y_tr  = generar_datos(20, semilla=42)   # 20 datos de entrenamiento
X_val, y_val = generar_datos(200, semilla=99)  # 200 datos de validacion
 
# =============================================================
# MLP para regresion (salida lineal, sin sigmoid en la capa final)
# =============================================================
def relu(z):     return np.maximum(0, z)
def d_relu(z):   return (z > 0).astype(float)
 
class MLPRegresion:
    def __init__(self, n_oculta, semilla=0):
        np.random.seed(semilla)
        escala = np.sqrt(2.0 / 1)
        self.W1 = np.random.randn(1, n_oculta) * escala
        self.b1 = np.zeros(n_oculta)
        self.W2 = np.random.randn(n_oculta, 1) * 0.1
        self.b2 = np.zeros(1)
        self.m = {'W1': np.zeros_like(self.W1), 'b1': np.zeros_like(self.b1),
                  'W2': np.zeros_like(self.W2), 'b2': np.zeros_like(self.b2)}
        self.v = {k: np.zeros_like(val) for k, val in self.m.items()}
        self.t = 0   # contador de pasos, para la correccion de sesgo de Adam
    
    def forward(self, X):
        self.X  = X
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = relu(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        return self.Z2
    
    def backward(self, dZ2, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        dW2 = self.A1.T @ dZ2            # dZ2 ya trae el 1/N de la pérdida media
        db2 = dZ2.sum(axis=0)
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * d_relu(self.Z1)
        dW1 = self.X.T @ dZ1
        db1 = dZ1.sum(axis=0)
        self.t += 1
        for nombre, grad in [('W1',dW1),('b1',db1),('W2',dW2),('b2',db2)]:
            self.m[nombre] = beta1*self.m[nombre] + (1-beta1)*grad
            self.v[nombre] = beta2*self.v[nombre] + (1-beta2)*grad**2
            m_hat = self.m[nombre] / (1 - beta1**self.t)
            v_hat = self.v[nombre] / (1 - beta2**self.t)
            getattr(self, nombre)[...] -= lr * m_hat / (np.sqrt(v_hat) + eps)
    
    def mse(self, X, y):
        return float(np.mean((self.forward(X) - y.reshape(-1,1))**2))
    
    def entrenar_epoch(self, X, y, lr):
        pred = self.forward(X)
        loss = float(np.mean((pred - y.reshape(-1,1))**2))
        grad = 2 * (pred - y.reshape(-1,1)) / len(X)
        self.backward(grad, lr)
        return loss
 
 
# =============================================================
# ENTRENAR TRES REDES: pequena, mediana y grande
# =============================================================
EPOCAS = 20000   # aumentado de 5000: con Adam y mas tiempo, el sobreajuste emerge con claridad
LR     = 0.01    # ajustado para Adam (antes 0.005, pensado para SGD)
 
modelos = [
    ('Modelo simple  (4 neuronas) ',  4),
    ('Modelo adecuado (16 neuronas)', 16),
    ('Modelo grande  (256 neuronas)', 256),
]
 
resultados = {}
for nombre, n_oc in modelos:
    red = MLPRegresion(n_oc, semilla=42)
    hist_tr, hist_val = [], []
    for ep in range(EPOCAS):
        lt = red.entrenar_epoch(X_tr, y_tr, LR)
        lv = red.mse(X_val, y_val)
        hist_tr.append(lt)
        hist_val.append(lv)
    resultados[nombre] = (hist_tr, hist_val)
 
# =============================================================
# MOSTRAR CURVAS DE APRENDIZAJE (formato tabla)
# =============================================================
def curva_ascii(hist_tr, hist_val, n_puntos=12):
    """Visualizacion ASCII de la curva de aprendizaje"""
    paso = len(hist_tr) // n_puntos
    print(f"  {'Epoca':>7}  {'Error train':>13}  {'Error val':>11}  Divergencia")
    print('  ' + '-'*55)
    for i in range(n_puntos):
        ep  = i * paso
        lt  = hist_tr[ep]
        lv  = hist_val[ep]
        div = lv / (lt + 1e-10)
        barra_tr = '*' * min(20, int(lt * 40))
        barra_vl = 'o' * min(20, int(lv * 40))
        if div > 2.0:
            señal = f'!!! OVERFITTING (val={div:.1f}x train)'
        elif lt > 0.15:
            señal = 'aun aprendiendo'
        else:
            señal = 'convergido'
        print(f'  {ep:>7}  {lt:>13.5f}  {lv:>11.5f}  {señal}')
 
print('='*60)
print('CURVAS DE APRENDIZAJE')
print('='*60)
 
for nombre, (hist_tr, hist_val) in resultados.items():
    print(f'\n{nombre}')
    curva_ascii(hist_tr, hist_val)
    mejor_val  = min(hist_val)
    epoca_mej  = hist_val.index(mejor_val)
    print(f'  -> Error train final:  {hist_tr[-1]:.5f}')
    print(f'  -> Error val final:    {hist_val[-1]:.5f}')
    print(f'  -> Mejor val (epoca {epoca_mej}): {mejor_val:.5f}')
    ratio = hist_val[-1] / (hist_tr[-1] + 1e-10)
    if ratio > 2.5:
        print(f'  -> DIAGNOSTICO: OVERFITTING grave (val es {ratio:.1f}x train)')
    elif hist_tr[-1] > 0.12:
        print(f'  -> DIAGNOSTICO: UNDERFITTING (error train alto)')
    else:
        print(f'  -> DIAGNOSTICO: Buen ajuste')
