# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import numpy as np
fuente = open('programa_O3_comparativa.py', encoding='utf-8').read()
exec(fuente.split('# COMPARATIVA')[0])   # datos, red y optimizadores (O-2 y O-3)
 
np.random.seed(42)
X, y = dos_lunas(300)
X_tr, y_tr   = X[:240], y[:240]
X_val, y_val = X[240:], y[240:]
 
valores_wd = [0.0, 0.001, 0.01, 0.1, 1.0]
 
print('=== EFECTO DEL WEIGHT DECAY EN ADAMW ===\n')
print(f"{'weight_decay':>14}  {'Loss train':>12}  {'Acc val':>10}  Diagnostico")
print('-' * 55)
 
for wd in valores_wd:
    params = inicializar(s=42)
    opt = AdamW(lr=0.01, wd=wd)
    for ep in range(300):
        pred, Z1, A1 = forward(X_tr, params)
        grads = backward(X_tr, y_tr, pred, Z1, A1, params)
        opt.paso(params, grads)
    pred_final, _, _ = forward(X_tr, params)
    loss = bce_loss(pred_final, y_tr)
    acc  = accuracy(X_val, y_val, params)
    diag = 'underfitting' if wd >= 1.0 else ('optimo' if 0.001<=wd<=0.1 else 'sin regularizar')
    print(f'{wd:>14.3f}  {loss:>12.5f}  {acc:>10.1%}  {diag}')
