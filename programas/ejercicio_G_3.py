# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import numpy as np

def softmax(x):
    e = np.exp(x - x.max()); return e / e.sum()

def generar_3_clases(n=80):
    X, y = [], []
    np.random.seed(1)
    for _ in range(n):
        for clase in range(3):
            img = np.zeros((10,10))
            ruido = np.random.randn(10,10)*0.1
            if clase == 0:   # horizontal
                f = np.random.randint(2, 7)
                img[f:f+2, 1:9] = 1.0
            elif clase == 1: # vertical
                c = np.random.randint(2, 7)
                img[1:9, c:c+2] = 1.0
            else:            # diagonal
                for i in range(8): img[i,i] = 1.0
            X.append(np.clip(img+ruido, 0, 1))
            y.append(clase)
    idx = np.random.permutation(len(X))
    return np.array(X)[idx], np.array(y)[idx]

X, y = generar_3_clases()
print(f'Dataset: {len(X)} imágenes, 3 clases')
print(f'Clase 0 (horiz): {sum(y==0)}  Clase 1 (vert): {sum(y==1)}  Clase 2 (diag): {sum(y==2)}')
print()
print('Cambios respecto al Programa 9B (clasificación binaria):')
print('  ANTES: Dense(16->1) + Sigmoid + BCE loss')
print('  AHORA: Dense(16->3) + Softmax + CE multiclase')
print()
print('Función de pérdida multiclase:')
print('  logits = red.forward(img)  # (3,)')
print('  probs  = softmax(logits)   # (3,) suman 1')
print('  loss   = -log(probs[clase_correcta])')
print()
logits_ej = np.array([2.0, 0.5, -0.3])
probs_ej  = softmax(logits_ej)
for i, (nombre, p) in enumerate(zip(['horiz','vert','diag'], probs_ej)):
    print(f'  P({nombre}) = {p:.4f}  {"<- clase predicha" if i==np.argmax(probs_ej) else ""}')
