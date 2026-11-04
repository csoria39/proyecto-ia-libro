# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import numpy as np
fuente = open('programa_R3_dropout.py', encoding='utf-8').read()
exec(fuente.split('# COMPARATIVA')[0])   # la clase MLPConDropout del Programa R-3
 
def gen(n, s): np.random.seed(s); X=np.sort(np.random.uniform(0,1,n)).reshape(-1,1); return X, np.sin(2*np.pi*X.ravel())+np.random.normal(0,0.25,n)
X_tr, y_tr   = gen(20, 42)
X_val, y_val = gen(200, 99)
 
configs = [
    ('Solo Dropout (p=0.3)',        0.3,  None),
    ('Solo Early Stopping',         0.0,  1000),   # paciencia como en R-4: no parar en una meseta
    ('Dropout + Early Stopping',    0.3,  1000),
]
 
print('=== COMBINANDO DROPOUT Y EARLY STOPPING ===\n')
print(f"{'Configuracion':>28}  {'Err Val Final':>14}  {'Mejor Err Val':>14}  {'Epoca parada':>13}")
resumen = []
print('-' * 75)
 
for nombre, p_drop, paciencia in configs:
    red = MLPConDropout(256, p_dropout=p_drop)
    mejor_val = float('inf')
    contador  = 0
    epoca_final = 5000
    mejores = None
    for ep in range(5000):
        red.entrenar_epoch(X_tr, y_tr, lr=0.01)
        err_val = red.evaluar(X_val, y_val)
        if paciencia is not None:
            if err_val < mejor_val - 1e-4:
                mejor_val = err_val; contador = 0
                mejores = [x.copy() for x in (red.W1, red.b1, red.W2, red.b2)]   # guardar los mejores pesos
            else:
                contador += 1
                if contador >= paciencia:
                    epoca_final = ep; break
        else:
            mejor_val = min(mejor_val, err_val)
    if mejores is not None:                       # early stopping: restaurar los mejores pesos
        red.W1, red.b1, red.W2, red.b2 = mejores
    err_final = red.evaluar(X_val, y_val)
    resumen.append((err_final, nombre))
    print(f'{nombre:>28}  {err_final:>14.5f}  {mejor_val:>14.5f}  {epoca_final:>13}')
 
print()
print(f'Mejor en esta ejecucion: {min(resumen)[1]} (err_val={min(resumen)[0]:.5f}).')
print('Que la combinacion gane depende del problema y de la semilla: con 20 puntos')
print('hay que repetir con varias semillas antes de sacar conclusiones.')
