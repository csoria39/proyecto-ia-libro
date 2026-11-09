# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import torch
import torch.nn as nn
 
# Tokenizador identico al Programa 5 NumPy
texto = 'hola mundo, aprendiendo inteligencia artificial'
vocab = sorted(set(texto))
V     = len(vocab)
c2i   = {c: i for i, c in enumerate(vocab)}
i2c   = dict(enumerate(vocab))
 
def codificar(t):   return torch.tensor([c2i[c] for c in t])
def decodificar(t): return ''.join(i2c[i.item()] for i in t)
 
# ============================================================
# nn.Embedding: tabla de busqueda diferenciable
# Es una matriz (V, DIM) cuyos valores se APRENDEN
# Se indexa igual que una lista: tabla[indice] -> vector
# ============================================================
DIM   = 8
tabla = nn.Embedding(V, DIM)
 
print(f'Vocabulario: {V} caracteres')
print(f'Tabla de embeddings: {tabla.weight.shape}')
print(f'Parametros aprendibles: {tabla.weight.numel()}')
print(f'  (= {V} tokens x {DIM} dim = {V * DIM})')
print()
 
# Uso: igual que indexar un array
tokens  = codificar('hola')
embs    = tabla(tokens)   # (4, 8)
print(f'Tokens de hola: {tokens.tolist()}')
print(f'Shape: {embs.shape}')
print(f'Embedding de h:')
print(f'  {embs[0].detach().numpy().round(3)}')
print()
 
# Similitud coseno entre dos embeddings
def sim_coseno(u, v):
    return (u @ v / (u.norm() * v.norm() + 1e-8)).item()
 
chars_test = ['h', 'o', 'l', 'a', ' ', 'i']
print('Similitudes coseno (aleatorias antes de entrenar):')
for c1, c2 in [('h','o'), ('h','h'), ('h',' ')]:
    e1 = tabla(torch.tensor(c2i[c1]))
    e2 = tabla(torch.tensor(c2i[c2]))
    print(f'  sim({repr(c1)}, {repr(c2)}) = {sim_coseno(e1, e2):.4f}')
 
print()
print('Tras el entrenamiento, caracteres en contextos')
print('similares tendran similitudes mas altas.')
print('Esto emerge de loss.backward() + optimizer.step().')
