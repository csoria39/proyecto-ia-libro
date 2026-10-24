# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import numpy as np
 
# =======================================================
# BLOQUES CONSTRUCTIVOS DEL MLP
# Cada bloque tiene métodos forward() y backward()
# =======================================================
 
class CapaDensa:
    """Capa totalmente conectada: salida = X @ W + b"""
    
    def __init__(self, n_ent, n_sal):
        # Inicialización He — buena para ReLU
        escala = np.sqrt(2.0 / n_ent)
        self.W = np.random.randn(n_ent, n_sal) * escala
        self.b = np.zeros(n_sal)
        # Cache: guardamos valores para backprop
        self._X = None
    
    def forward(self, X):
        self._X = X                      # guardamos para backprop
        return X @ self.W + self.b
    
    def backward(self, dZ, lr):
        m   = self._X.shape[0]           # tamaño del lote
        dW  = self._X.T @ dZ / m         # gradiente de W
        db  = dZ.mean(axis=0)            # gradiente de b
        dX  = dZ @ self.W.T              # gradiente hacia atrás
        self.W -= lr * dW                # actualizar W
        self.b -= lr * db                # actualizar b
        return dX
 
 
class ReLU:
    """Activación ReLU con su derivada para backprop"""
    def __init__(self): self._Z = None
    def forward(self, Z):
        self._Z = Z
        return np.maximum(0, Z)
    def backward(self, dA, lr=None):
        return dA * (self._Z > 0)        # derivada es 0 ó 1
 
 
class Sigmoid:
    """Activación sigmoid con su derivada"""
    def __init__(self): self._A = None
    def forward(self, Z):
        self._A = 1.0 / (1.0 + np.exp(-np.clip(Z, -500, 500)))
        return self._A
    def backward(self, dA, lr=None):
        s = self._A
        return dA * s * (1.0 - s)
class RedNeuronal:
    """
    MLP genérico: acepta cualquier secuencia de capas.
    Ejemplo de uso:
      red = RedNeuronal([
          CapaDensa(2, 4), ReLU(),
          CapaDensa(4, 1), Sigmoid()
      ])
    """
    
    def __init__(self, capas):
        self.capas = capas
    
    def predecir(self, X):
        """Paso forward completo"""
        salida = X
        for capa in self.capas:
            salida = capa.forward(salida)
        return salida
    
    def _backward(self, grad, lr):
        """Backpropagation: propaga el gradiente hacia atrás"""
        for capa in reversed(self.capas):
            grad = capa.backward(grad, lr)
    
    def entrenar(self, X, y, epocas=2000, lr=0.1, verbose=True):
        """Bucle de entrenamiento completo"""
        y = y.reshape(-1, 1)
        historico = []
        
        for epoca in range(epocas):
            # --- Forward ---
            pred  = self.predecir(X)
            # --- Pérdida MSE ---
            loss  = float(np.mean((pred - y) ** 2))
            historico.append(loss)
            # --- Gradiente de MSE respecto a la predicción ---
            dPred = 2.0 * (pred - y) / len(X)
            # --- Backward ---
            self._backward(dPred, lr)
            
            if verbose and (epoca % 500 == 0 or epoca == epocas - 1):
                print(f'  Época {epoca:5d}: pérdida = {loss:.6f}')
        
        return historico
