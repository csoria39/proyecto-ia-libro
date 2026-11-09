# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
# Fragmento 16.3 — Pasar de nats por token a bits por carácter
import numpy as np
fuente = open('programa_8bis_gpt2_real.py', encoding='utf-8').read()
exec(fuente.split('# EXPERIMENTOS')[0])

texto = ('Neural networks learn by adjusting their weights. During training, the network makes a '
         'prediction, measures its error with a loss function, and uses backpropagation to compute '
         'how each weight should change to reduce that error.')
ids = codificar(texto)
lp = np.log(softmax(forward(np.array([ids]))[0][0, :-1].astype(np.float64)))
nats = -lp[np.arange(len(ids) - 1), ids[1:]].sum()       # suma de las 43 pérdidas
chars = len(texto) - len(decodificar(ids[:1]))           # caracteres predichos
print(f'{len(ids) - 1} tokens predichos, {chars} caracteres')
print(f'{nats / (len(ids) - 1):.3f} nats/token  =  {nats / chars:.3f} nats/carácter  =  '
      f'{nats / chars / np.log(2):.2f} bits/carácter')
print(f'Un token ocupa de media {chars / (len(ids) - 1):.1f} caracteres')
