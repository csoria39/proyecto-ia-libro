# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import numpy as np
 
def sigmoid(z): return 1/(1+np.exp(-z))
 
# Parámetros escogidos a mano para OR
# Lógica: queremos sigmoid(w+b) > 0.5 cuando solo hay 1 entrada activa
# sigmoid(z) > 0.5  <=>  z > 0
# Para OR: necesitamos  w1*1 + w2*0 + b > 0  y  w1*0 + w2*0 + b < 0
# Es decir:  w1 + b > 0  y  b < 0
# Con w1=w2=8, b=-4:  8-4=4>0 ✓  y  -4<0 ✓
 
w1, w2, b = 8.0, 8.0, -4.0
 
X = [[0,0],[0,1],[1,0],[1,1]]
y = [0,   1,   1,   1  ]
 
print('Puerta OR — parámetros ajustados manualmente')
print(f'w1={w1}, w2={w2}, sesgo={b}')
print()
print(f"{'Entradas':^12}  {'z':>8}  {'σ(z)':>10}  {'Esperado':>8}  OK")
print('-' * 50)
correcto = 0
for xi, yi in zip(X, y):
    z   = xi[0]*w1 + xi[1]*w2 + b
    sal = sigmoid(z)
    ok  = round(sal) == yi
    correcto += ok
    print(f'{str(xi):^12}  {z:>8.1f}  {sal:>10.4f}  {yi:>8}  {"✓" if ok else "✗"}')
print(f'\nPrecisión: {correcto}/4')
