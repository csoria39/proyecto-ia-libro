# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import numpy as np
 
entradas = np.array([0.5,  0.8])
pesos    = np.array([0.3, -0.4])
sesgo    = 0.2
 
def sigmoid(z): return 1 / (1 + np.exp(-z))
 
# --- Cálculo paso a paso ---
p1 = entradas[0] * pesos[0]    # 0.5 x 0.30 =  0.15
p2 = entradas[1] * pesos[1]    # 0.8 x (-0.4) = -0.32
z  = p1 + p2 + sesgo           # 0.15 + (-0.32) + 0.20 = 0.03
sal = sigmoid(z)
 
print('=== CÁLCULO DETALLADO ===')
print(f'  x1.w1 = {entradas[0]} x {pesos[0]}  = {p1:.4f}')
print(f'  x2.w2 = {entradas[1]} x {pesos[1]} = {p2:.4f}')
print(f'  sesgo =                   {sesgo:.4f}')
print(f'  z     = {p1:.4f} + ({p2:.4f}) + {sesgo:.4f} = {z:.4f}')
print(f'  σ({z:.4f}) = 1/(1+e^{-z:.4f}) = {sal:.6f}')
print()
print(f'Como z={z:.4f} > 0, la salida ({sal:.4f}) > 0.5 ✓')
