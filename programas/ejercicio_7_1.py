# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import numpy as np
fuente = open('programa_R1_overfitting.py', encoding='utf-8').read()
exec(fuente.split('# ENTRENAR')[0])   # la clase MLPRegresion del Programa R-1
 
def gen(n, s): np.random.seed(s); X=np.sort(np.random.uniform(0,1,n)).reshape(-1,1); return X, np.sin(2*np.pi*X.ravel())+np.random.normal(0,0.25,n)
X_tr, y_tr   = gen(20, 42)
X_val, y_val = gen(200, 99)
 
tamanos = [1, 2, 4, 8, 16, 32, 64, 128, 256]
 
print('=== CURVA DE COMPLEJIDAD (sesgo-varianza) ===\n')
print(f"{'Neuronas':>10}  {'Err. Train':>12}  {'Err. Val':>10}  Barra Val")
print('-' * 55)
 
resultados = []
for n_oc in tamanos:
    red = MLPRegresion(n_oc, semilla=42)
    for _ in range(20000):   # aumentado de 3000: con Adam hace falta mas tiempo
        red.entrenar_epoch(X_tr, y_tr, lr=0.01)
    err_tr  = red.mse(X_tr, y_tr)
    err_val = red.mse(X_val, y_val)
    resultados.append((n_oc, err_tr, err_val))
    barra = '#' * min(40, int(err_val * 80))
    print(f'{n_oc:>10}  {err_tr:>12.5f}  {err_val:>10.5f}  {barra}')
 
mejor = min(resultados, key=lambda r: r[2])
print(f'\nMejor configuracion: {mejor[0]} neuronas (err_val={mejor[2]:.5f})')
print('Antes de ese punto: underfitting. Despues: overfitting creciente.')
