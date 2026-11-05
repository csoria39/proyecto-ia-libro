# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import numpy as np
 
# =======================================================
# TOKENIZADOR A NIVEL DE CARÁCTER + EMBEDDINGS
# =======================================================
 
class TokenizadorCaracter:
    """Convierte texto en secuencias de índices enteros"""
    
    def __init__(self, texto):
        # El vocabulario son los caracteres únicos del texto
        self.vocab = sorted(set(texto))
        self.V = len(self.vocab)
        self.c2i = {c: i for i, c in enumerate(self.vocab)}
        self.i2c = {i: c for i, c in enumerate(self.vocab)}
        print(f'Vocabulario: {self.V} caracteres únicos')
    
    def codificar(self, texto):
        """Texto -> lista de enteros. OJO: un carácter que no esté en el
        vocabulario se descarta sin avisar (el BPE del Capítulo 9 lo evita)."""
        return [self.c2i[c] for c in texto if c in self.c2i]
    
    def decodificar(self, indices):
        """Lista de enteros -> texto"""
        return ''.join(self.i2c[i] for i in indices)
 
 
class TablaEmbeddings:
    """
    Tabla de vectores densos para cada token.
    Cada token tiene asignado un vector de dimensión 'dim'.
    Los vectores se APRENDEN durante el entrenamiento.
    """
    
    def __init__(self, V, dim):
        # Inicialización con valores pequeños aleatorios
        self.E = np.random.randn(V, dim) * 0.01
        self.V   = V
        self.dim = dim
    
    def obtener(self, indices):
        """
        Lookup: dado un índice (o lista), devuelve su vector.
        Equivalente a: E[indices]
        """
        return self.E[indices]
    
    def similitud(self, i, j):
        """Coseno entre dos vectores de embedding"""
        u = self.E[i] / (np.linalg.norm(self.E[i]) + 1e-8)
        v = self.E[j] / (np.linalg.norm(self.E[j]) + 1e-8)
        return float(np.dot(u, v))
 
 
# -- Demostración ---------------------------------------
texto_ejemplo = 'hola mundo, aprendiendo inteligencia artificial'
tok = TokenizadorCaracter(texto_ejemplo)
 
# Codificación y decodificación
fragmento = 'hola'
codificado   = tok.codificar(fragmento)
decodificado = tok.decodificar(codificado)
print(f'\nOriginal:    "{fragmento}"')
print(f'Codificado:  {codificado}')
print(f'Decodificado:"{decodificado}"')
 
# Crear tabla de embeddings (dim=6 para visualización)
np.random.seed(7)
emb = TablaEmbeddings(V=tok.V, dim=6)
 
# Mostrar embeddings de algunos caracteres
print(f'\nEmbeddings iniciales (antes de entrenar):')
for char in ['h', 'o', 'l', 'a', ' ']:
    idx = tok.c2i[char]
    vec = emb.obtener(idx)
    print(f'  [{repr(char)}] índice={idx:2d}  {vec.round(3)}')
 
# Nota: después del entrenamiento, caracteres con
# contextos similares tendrán vectores más parecidos
print('\nNota: Los embeddings se aprenden durante el entrenamiento.')
print('Después, caracteres en contextos similares tienen')
print('vectores matemáticamente más cercanos.')
