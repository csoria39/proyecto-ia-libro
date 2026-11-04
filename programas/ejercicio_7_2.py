# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import numpy as np
fuente = open('programa_R1_overfitting.py', encoding='utf-8').read()
exec(fuente.split('# ENTRENAR')[0])   # la clase MLPRegresion del Programa R-1
 
def gen(n, s): np.random.seed(s); X=np.sort(np.random.uniform(0,1,n)).reshape(-1,1); return X, np.sin(2*np.pi*X.ravel())+np.random.normal(0,0.25,n)
X_val, y_val = gen(200, 99)
 
tamanos_dataset = [10, 20, 50, 100, 500]
 
print('=== EFECTO DEL TAMAÑO DEL DATASET (red fija: 256 neuronas) ===\n')
print(f"{'N ejemplos':>11}  {'Err Train':>10}  {'Err Val':>10}  {'Ratio Val/Train':>16}")
print('-' * 55)
 
for n in tamanos_dataset:
    X_tr, y_tr = gen(n, 42)
    red = MLPRegresion(256, semilla=42)
    epocas = max(20000, 400000 // n)  # mas epocas si hay menos datos; suelo alto para Adam
    for _ in range(epocas):
        red.entrenar_epoch(X_tr, y_tr, lr=0.01)
    err_tr  = red.mse(X_tr, y_tr)
    err_val = red.mse(X_val, y_val)
    ratio   = err_val / (err_tr + 1e-10)
    print(f'{n:>11}  {err_tr:>10.5f}  {err_val:>10.5f}  {ratio:>16.1f}x')
 
print()
print('Con pocos datos (n=10), la misma red memoriza perfectamente')
print('pero generaliza fatal. Con muchos datos (n=500), ya no puede')
print('memorizar cada punto y se ve forzada a generalizar mejor.')
