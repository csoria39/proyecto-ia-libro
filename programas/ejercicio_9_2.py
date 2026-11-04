# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import numpy as np
fuente = open('programa_BPE2_codificador.py', encoding='utf-8').read()
exec(fuente.split('# Entrenar con nuestro corpus')[0])   # BPE-1 y BPE-2
 
CORPUS = '''
la inteligencia artificial aprende de los datos de entrenamiento.
los modelos de lenguaje aprenden a predecir el siguiente token.
las redes neuronales aprenden representaciones de los datos.
'''
 
n_fusiones_lista = [5, 10, 20, 40, 80]
n_chars = sum(len(w) for w in CORPUS.split())
 
print(f'Caracteres totales en el corpus: {n_chars}\n')
print(f"{'N fusiones':>11}  {'Tokens BPE':>11}  {'Compresion':>11}  {'Mejora vs anterior':>19}")
print('-' * 60)
 
anterior = n_chars
for nf in n_fusiones_lista:
    reglas, vocab = entrenar_bpe(CORPUS, num_fusiones=nf, verbose=False)
    n_tokens = len(codificar_bpe(CORPUS, reglas))
    compresion = n_chars / n_tokens
    mejora = (anterior - n_tokens) / anterior * 100
    print(f'{nf:>11}  {n_tokens:>11}  {compresion:>10.2f}x  {mejora:>18.1f}%')
    anterior = n_tokens
 
print()
print('La compresion mejora con cada fusion, pero la GANANCIA marginal')
print('decrece: las primeras fusiones (patrones frecuentes) aportan')
print('mucho mas que las ultimas (casos cada vez mas raros).')
