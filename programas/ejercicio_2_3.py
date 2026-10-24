# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import numpy as np
 
def sigmoid(z): return 1/(1+np.exp(-z))
 
peso   = 1.0
x_fijo = 0.0
 
print('Neurona con 1 entrada: w=1.0, x=0 (fija)')
print(f"{'Sesgo b':>10}  {'z=x*w+b':>10}  {'σ(z)':>10}  Estado")
print('-' * 50)
 
for b in range(-5, 6):
    z = x_fijo * peso + b
    s = sigmoid(z)
    if s < 0.12:   estado = 'prácticamente apagada (≈0)'
    elif s < 0.40: estado = 'inhibida'
    elif s < 0.60: estado = 'neutral (≈0.5)'
    elif s < 0.88: estado = 'activa'
    else:          estado = 'prácticamente encendida (≈1)'
    print(f'{b:>10.0f}  {z:>10.1f}  {s:>10.4f}  {estado}')
 
print()
print('Conclusión: el sesgo controla el umbral de activación.')
print('Sesgo negativo -> la neurona necesita entradas más fuertes')
print('Sesgo positivo -> la neurona se activa incluso con entradas débiles')
