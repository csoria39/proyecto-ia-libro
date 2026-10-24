# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import numpy as np
 
class NeuronaFlex:
    def __init__(self, n, activacion='sigmoid', semilla=42):
        np.random.seed(semilla)
        self.W = np.random.randn(n) * 0.5
        self.b = 0.1
        self.activacion = activacion
    
    def calcular(self, x):
        z = np.dot(np.array(x), self.W) + self.b
        if self.activacion == 'sigmoid':
            return 1/(1+np.exp(-z)), z
        elif self.activacion == 'relu':
            return max(0.0, z), z
        elif self.activacion == 'tanh':
            return np.tanh(z), z
 
n_sig = NeuronaFlex(3, 'sigmoid')
n_rel = NeuronaFlex(3, 'relu')
 
casos = [[1.0,1.0,1.0], [-1.0,-1.0,-1.0], [0.0,0.0,0.0]]
print(f"{'Entradas':>22}  {'Sigmoid':>10}  {'ReLU':>10}  {'z':>8}")
print('-' * 57)
for x in casos:
    s, z = n_sig.calcular(x)
    r, _ = n_rel.calcular(x)
    print(f'{str(x):>22}  {s:>10.4f}  {r:>10.4f}  {z:>8.4f}')
 
print()
print('Diferencias clave:')
print('* Sigmoid: acotada en (0,1), nunca exactamente 0')
print('* ReLU:    puede ser 0 exacto (neurona apagada)')
print('* ReLU:    sin acotación superior')
