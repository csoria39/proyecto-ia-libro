# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import numpy as np
 
def sigmoid(z): return 1/(1+np.exp(-z))
def d_sig(z): s=sigmoid(z); return s*(1-s)
 
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
 
tablas = {
    'AND':  np.array([0, 0, 0, 1], dtype=float),
    'OR':   np.array([0, 1, 1, 1], dtype=float),
    'NAND': np.array([1, 1, 1, 0], dtype=float),
}
 
for nombre, y in tablas.items():
    np.random.seed(1)
    w = np.random.randn(2) * 0.3
    b = 0.0
    for _ in range(8000):
        for xi, yi in zip(X, y):
            z = np.dot(xi, w) + b
            d = (yi - sigmoid(z)) * d_sig(z)
            w += 0.5 * d * xi
            b += 0.5 * d
    print(f'{nombre}: w=[{w[0]:.2f}, {w[1]:.2f}]  b={b:.2f}', end='   ')
    pred = [round(float(sigmoid(np.dot(xi,w)+b))) for xi in X]
    print(f'preds={pred}  {"✓" if pred == list(y.astype(int)) else "✗"}')
 
print()
print('Observación: NAND tiene pesos negativos y sesgo positivo.')
print('Esto tiene sentido: NAND activa la salida por defecto')
print('y la inhibe cuando recibe entradas positivas fuertes.')
