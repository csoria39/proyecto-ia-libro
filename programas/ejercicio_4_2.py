# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import numpy as np
 
def sigmoid(z): return 1/(1+np.exp(-z))
def d_sig(z): s=sigmoid(z); return s*(1-s)
 
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([0, 0, 0, 1], dtype=float)
 
def perdida(w, b):
    return sum((y[i]-sigmoid(np.dot(X[i],w)+b))**2 for i in range(4))/4
 
print(f"{'lr':>8}  {'Épocas hasta conv.':>20}  {'Pérdida final':>15}  Estado")
print('-' * 60)
 
for lr in [0.001, 0.01, 0.1, 0.5, 1.0, 5.0]:
    np.random.seed(42)
    w = np.random.randn(2) * 0.3; b = 0.0
    ep_conv = None
    for ep in range(15000):
        for xi, yi in zip(X, y):
            z = np.dot(xi, w) + b
            d = (yi - sigmoid(z)) * d_sig(z)
            w += lr * d * xi; b += lr * d
        if perdida(w, b) < 0.005 and ep_conv is None:
            ep_conv = ep + 1
    L = perdida(w, b)
    ep_str  = str(ep_conv) if ep_conv else '>15000'
    estado  = 'converge' if L < 0.01 else 'NO converge'
    print(f'{lr:>8.3f}  {ep_str:>20}  {L:>15.6f}  {estado}')
