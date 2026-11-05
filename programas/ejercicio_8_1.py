# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import numpy as np
 
class TokenizadorCaracter:
    def __init__(self, texto):
        self.vocab = sorted(set(texto))
        self.V     = len(self.vocab)
        self.c2i   = {c:i for i,c in enumerate(self.vocab)}
        self.i2c   = dict(enumerate(self.vocab))
    def codificar(self, t):  return [self.c2i[c] for c in t if c in self.c2i]   # descarta lo desconocido
    def decodificar(self, l): return ''.join(self.i2c[i] for i in l)
 
# Cambia por tu propio nombre
nombre = 'Alan Turing'
 
tok = TokenizadorCaracter(nombre)
print(f'Nombre: "{nombre}"')
print(f'Vocabulario ({tok.V} caracteres): {tok.vocab}')
print()
 
cod = tok.codificar(nombre)
dec = tok.decodificar(cod)
print(f'Codificado:   {cod}')
print(f'Decodificado: "{dec}"')
print(f'Reversible:   {nombre == dec}')
print()
print('Índice de cada carácter:')
for c, i in tok.c2i.items():
    print(f'  {repr(c):>6}  ->  índice {i}')
