# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
from collections import Counter
 
def texto_a_vocab(palabras_freq):
    return {tuple(list(p) + ['</w>']): f for p, f in palabras_freq.items()}
 
def contar_pares(vocab):
    pares = Counter()
    for palabra, freq in vocab.items():
        for i in range(len(palabra)-1):
            pares[(palabra[i], palabra[i+1])] += freq
    return pares
 
def fusionar(vocab, par):
    a, b = par; nuevo = {}
    for palabra, freq in vocab.items():
        nueva_pal, i = [], 0
        while i < len(palabra):
            if i<len(palabra)-1 and palabra[i]==a and palabra[i+1]==b:
                nueva_pal.append(a+b); i += 2
            else:
                nueva_pal.append(palabra[i]); i += 1
        nuevo[tuple(nueva_pal)] = freq
    return nuevo
 
corpus = {'bajo': 6, 'bajar': 3, 'caja': 4, 'cajon': 2}
vocab  = texto_a_vocab(corpus)
 
print('=== BPE A MANO: 2 ITERACIONES ===\n')
print('Vocabulario inicial:')
for p, f in vocab.items(): print(f'  {p} x{f}')
 
for it in range(1, 3):
    pares = contar_pares(vocab)
    mejor = max(pares, key=pares.get)
    print(f'\n--- Iteracion {it} ---')
    print(f'Pares mas frecuentes: {pares.most_common(3)}')
    print(f'Fusionando: {mejor} (frecuencia {pares[mejor]})')
    vocab = fusionar(vocab, mejor)
    print('Vocabulario tras fusion:')
    for p, f in vocab.items(): print(f'  {p} x{f}')
