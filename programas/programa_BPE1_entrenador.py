# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
from collections import Counter
import re
 
# ================================================================
# ENTRENADOR BPE (Byte Pair Encoding)
# Aprende las reglas de fusion optimas sobre un corpus de texto.
# ================================================================
 
def texto_a_vocab_inicial(texto):
    '''
    Convierte el texto en un vocabulario de frecuencias.
    Cada palabra se representa como una tupla de caracteres
    mas el marcador especial </w> (end of word).
    '''
    contador = Counter()
    for palabra in texto.lower().split():
        # Eliminar puntuacion basica
        # Simplificación didáctica: mantenemos solo letras minúsculas
        # Un tokenizador real trabaja sobre bytes UTF-8 completos.
        palabra = re.sub(r'[^a-záéíóúüñ]', '', palabra.lower())
        if palabra:
            contador[tuple(list(palabra) + ['</w>'])] += 1
    return dict(contador)
 
 
def contar_pares(vocab):
    '''Cuenta la frecuencia de todos los pares adyacentes.'''
    pares = Counter()
    for palabra, freq in vocab.items():
        for i in range(len(palabra) - 1):
            pares[(palabra[i], palabra[i+1])] += freq
    return pares
 
 
def fusionar_par(vocab, par):
    '''Fusiona el par dado en todas las palabras del vocabulario.'''
    a, b   = par
    nuevo  = {}
    for palabra, freq in vocab.items():
        nueva_pal = []
        i = 0
        while i < len(palabra):
            if i < len(palabra) - 1 and palabra[i] == a and palabra[i+1] == b:
                nueva_pal.append(a + b)   # fusion!
                i += 2
            else:
                nueva_pal.append(palabra[i])
                i += 1
        nuevo[tuple(nueva_pal)] = freq
    return nuevo
 
 
def entrenar_bpe(corpus, num_fusiones, verbose=True):
    '''
    Entrena el tokenizador BPE.
    Devuelve la lista de reglas de fusion aprendidas.
    '''
    vocab       = texto_a_vocab_inicial(corpus)
    vocab_chars = set(c for pal in vocab for c in pal)
    reglas      = []
    
    if verbose:
        print(f'Vocabulario inicial: {len(vocab_chars)} simbolos')
        print(f'Palabras unicas:     {len(vocab)}')
        print(f'Tokens iniciales en corpus: {sum(len(k)*v for k,v in vocab.items())}')
        print()
        print(f"{'Fusion':>7}  {'Par':>25}  {'Frec':>6}  {'Vocab size':>11}")
        print('-' * 55)
    
    for i in range(num_fusiones):
        pares = contar_pares(vocab)
        if not pares:
            break
        
        # Elegir el par mas frecuente
        mejor_par  = max(pares, key=pares.get)
        mejor_freq = pares[mejor_par]
        
        # Fusionar
        vocab  = fusionar_par(vocab, mejor_par)
        reglas.append(mejor_par)
        
        # El nuevo token es la concatenacion del par
        token_nuevo = mejor_par[0] + mejor_par[1]
        vocab_chars.add(token_nuevo)
        
        if verbose and (i < 10 or (i+1) % 10 == 0):
            print(f"  {i+1:>5}  {str(mejor_par):>25}  ",
                  f"{mejor_freq:>6}  {len(vocab_chars):>11}")
    
    if verbose:
        tokens_final = sum(len(k)*v for k,v in vocab.items())
        print(f'\nVocabulario final:   {len(vocab_chars)} simbolos')
        print(f'Tokens en corpus:    {tokens_final}')
        print(f'Reglas aprendidas:   {len(reglas)}')
    
    return reglas, vocab_chars
 
 
# ================================================================
# CORPUS DE ENTRENAMIENTO
# ================================================================
CORPUS = '''
la inteligencia artificial aprende de los datos de entrenamiento.
los modelos de lenguaje aprenden a predecir el siguiente token.
las redes neuronales aprenden representaciones de los datos.
el aprendizaje automatico permite que las maquinas aprendan.
la tokenizacion divide el texto en unidades llamadas tokens.
el vocabulario define el conjunto de tokens posibles.
los transformers usan atencion para procesar secuencias.
el entrenamiento ajusta los pesos de la red neuronal.
la inteligencia artificial transforma la tecnologia moderna.
aprender a programar inteligencia artificial es fascinante.
'''
 
print('=== ENTRENANDO TOKENIZADOR BPE ===')
print('Corpus de', len(CORPUS.split()), 'palabras\n')
 
reglas, vocab = entrenar_bpe(CORPUS, num_fusiones=30, verbose=True)
 
# Mostrar los tokens aprendidos mas interesantes
print('\nTokens aprendidos (subpalabras emergentes):')
tokens_subpal = [t for t in sorted(vocab) if len(t)>2 and '</w>' not in t]
print('  ' + '  '.join(sorted(tokens_subpal, key=len, reverse=True)[:20]))
