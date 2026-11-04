# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import re
fuente = open('programa_BPE1_entrenador.py', encoding='utf-8').read()
exec(fuente.split("print('=== ENTRENANDO")[0])   # entrenar_bpe() y CORPUS (BPE-1)
 
# ================================================================
# CODIFICADOR BPE
# Aplica las reglas de fusion aprendidas a texto nuevo.
# ================================================================
 
def codificar_bpe(texto, reglas):
    '''
    Convierte texto en una lista de tokens BPE.
    Aplica las reglas aprendidas en el mismo orden en que
    fueron aprendidas durante el entrenamiento.
    '''
    tokens_totales = []
    for palabra in texto.lower().split():
        palabra = re.sub(r'[^a-záéíóúüñ]', '', palabra)   # la misma normalización que BPE-1
        if not palabra:
            continue
        # Empezar con caracteres individuales
        chars = list(palabra) + ['</w>']
        # Aplicar reglas en orden de aprendizaje
        for (a, b) in reglas:
            nueva = []
            i = 0
            while i < len(chars):
                if i < len(chars)-1 and chars[i]==a and chars[i+1]==b:
                    nueva.append(a + b)
                    i += 2
                else:
                    nueva.append(chars[i])
                    i += 1
            chars = nueva
        tokens_totales.extend(chars)
    return tokens_totales
 
 
def decodificar_bpe(tokens):
    '''Convierte tokens BPE de vuelta a texto (normalizado).
    Se unen los tokens tal cual y cada </w> marca el final de una palabra.
    Recupera las palabras normalizadas (minúsculas, sin cifras ni puntuación),
    no el texto original letra a letra.'''
    return ''.join(tokens).replace('</w>', ' ').strip()
 
 
def comparar_tokenizaciones(frase, reglas):
    '''Compara tokenizacion por caracter vs BPE.'''
    # Char-level
    tokens_char = list(frase.replace(' ', '_'))
    # BPE
    tokens_bpe  = codificar_bpe(frase, reglas)
    tokens_bpe_limpios = [t.replace('</w>', '+') for t in tokens_bpe]
    print(f'  Frase: "{frase}"')
    print(f'  Char: {len(tokens_char):3d} tokens  {tokens_char[:15]}...')
    print(f'  BPE:  {len(tokens_bpe):3d} tokens  {tokens_bpe_limpios}')
    ratio = len(tokens_char) / max(len(tokens_bpe), 1)
    print(f'  BPE es {ratio:.1f}x mas eficiente que char-level')
    print()
 
 
# Entrenar con nuestro corpus
reglas, vocab = entrenar_bpe(CORPUS, num_fusiones=30, verbose=False)
 
print('=== TOKENIZACION BPE EN ACCION ===')
print()
 
frases = [
    'la inteligencia artificial aprende',
    'aprendizaje automatico',
    'redes neuronales profundas',
    'tokenizacion de texto',
]
for frase in frases:
    comparar_tokenizaciones(frase, reglas)
 
# Manejar palabras fuera del vocabulario
print('--- Manejo de palabras desconocidas ---')
frase_oov = 'supercalifragilistico'
tokens_oov = codificar_bpe(frase_oov, reglas)
print(f'  Palabra rara: "{frase_oov}"')
print(f'  BPE la descompone: {[t.replace("</w>","+") for t in tokens_oov]}')
print(f'  Sin OOV mientras sus letras esten en el corpus: cae hasta nivel caracter')

# Ida y vuelta: codificar y decodificar
print('\n--- Ida y vuelta (texto normalizado) ---')
for f in ['hola mundo', 'El niño aprende, 2026.']:
    print(f'  {f!r} -> {decodificar_bpe(codificar_bpe(f, reglas))!r}')
print('  Se recuperan las palabras normalizadas, no el texto literal:')
print('  mayusculas, cifras y puntuacion se pierden (GPT-2 lo evita con BPE por bytes).')
 
# Estadisticas de compresion
print('\n--- Compresion del corpus completo ---')
tokens_char = sum(len(w) for w in CORPUS.split())
tokens_bpe  = len(codificar_bpe(CORPUS, reglas))
print(f'  Chars en corpus:  {tokens_char}')
print(f'  Tokens BPE:       {tokens_bpe}')
print(f'  Ratio de compresion: {tokens_char/tokens_bpe:.2f}x')
