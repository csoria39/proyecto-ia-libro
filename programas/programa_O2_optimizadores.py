# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import numpy as np
 
# =============================================================
# FAMILIA DE OPTIMIZADORES — IMPLEMENTACION DESDE CERO
# Todos implementan la misma interfaz: paso(params, grads)
# =============================================================
 
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
    def paso(self, params, grads):
        for k in params:
            params[k] -= self.lr * grads[k]
 
 
class SGDMomentum:
    def __init__(self, lr=0.01, beta=0.9):
        self.lr   = lr
        self.beta = beta
        self.v    = {}
    def paso(self, params, grads):
        for k in params:
            if k not in self.v:
                self.v[k] = np.zeros_like(params[k])
            self.v[k]  = self.beta * self.v[k] + grads[k]
            params[k] -= self.lr * self.v[k]
 
 
class RMSprop:
    def __init__(self, lr=0.001, beta=0.99, eps=1e-8):
        self.lr   = lr
        self.beta = beta
        self.eps  = eps
        self.s    = {}
    def paso(self, params, grads):
        for k in params:
            if k not in self.s:
                self.s[k] = np.zeros_like(params[k])
            g          = grads[k]
            self.s[k]  = self.beta * self.s[k] + (1-self.beta) * g**2
            params[k] -= self.lr * g / (np.sqrt(self.s[k]) + self.eps)
 
 
class Adam:
    """
    El optimizador mas usado en Deep Learning.
    Combina Momentum (primer momento) + RMSprop (segundo momento)
    con correccion de sesgo para los primeros pasos.
    """
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr    = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self.m = {}    # primer momento (media del gradiente)
        self.v = {}    # segundo momento (media del gradiente^2)
        self.t = 0     # contador de pasos (para correccion de sesgo)
    
    def paso(self, params, grads):
        self.t += 1
        for k in params:
            if k not in self.m:
                self.m[k] = np.zeros_like(params[k])
                self.v[k] = np.zeros_like(params[k])
            g = grads[k]
            # Actualizar momentos
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * g**2
            # Correccion de sesgo
            m_hat = self.m[k] / (1 - self.beta1**self.t)
            v_hat = self.v[k] / (1 - self.beta2**self.t)
            # Actualizacion final
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
 
 
class AdamW(Adam):
    """
    Adam con weight decay desacoplado.
    Estandar para entrenar Transformers y LLMs.
    """
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.01):
        super().__init__(lr, beta1, beta2, eps)
        self.wd = wd
    
    def paso(self, params, grads):
        self.t += 1
        for k in params:
            if k not in self.m:
                self.m[k] = np.zeros_like(params[k])
                self.v[k] = np.zeros_like(params[k])
            # Weight decay desacoplado (ANTES del gradiente)
            params[k] *= (1 - self.lr * self.wd)
            g = grads[k]
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * g**2
            m_hat = self.m[k] / (1 - self.beta1**self.t)
            v_hat = self.v[k] / (1 - self.beta2**self.t)
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
 
 
print('Optimizadores implementados correctamente.')
print('Clases disponibles: SGD, SGDMomentum, RMSprop, Adam, AdamW')
print()
# Verificacion: todos tienen la misma interfaz
opts = [SGD(0.01), SGDMomentum(0.01), RMSprop(0.001), Adam(0.001), AdamW(0.001)]
params_test = {'W': np.array([1.0, -0.5, 2.0])}
grads_test  = {'W': np.array([0.1, -0.2, 0.3])}
for opt in opts:
    p = {k: v.copy() for k,v in params_test.items()}
    opt.paso(p, grads_test)
    print(f'{opt.__class__.__name__:>14}: W -> {p["W"].round(5)}')
