# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import numpy as np
fuente = open('programa_BPE2_codificador.py', encoding='utf-8').read()
exec(fuente.split('# Entrenar con nuestro corpus')[0])   # BPE-1 y BPE-2
 
# Simulamos las dos situaciones lingüísticas
CORPUS_TIPO_INGLES = '''
la red es buena la red aprende el dato es bueno
el dato fluye la red usa el dato la red es util
'''
 
CORPUS_TIPO_ALEMAN = '''
lareduronal esbuenaparaaprender eldatoesimportanteparalared
elflujodedatos esutilparalaredneuronalartificial
'''
 
for nombre, corpus in [('Tipo INGLES (palabras cortas)', CORPUS_TIPO_INGLES),
                       ('Tipo ALEMAN (compuestos largos)', CORPUS_TIPO_ALEMAN)]:
    reglas, vocab = entrenar_bpe(corpus, num_fusiones=20, verbose=False)
    n_chars  = len(corpus.replace(chr(10), '').replace(' ', ''))
    n_tokens = len(codificar_bpe(corpus, reglas))
    fertilidad = n_tokens / len(corpus.split())
    print(f'{nombre}:')
    print(f'  Caracteres: {n_chars}  Tokens: {n_tokens}  '
          f'Fertilidad: {fertilidad:.2f} tokens/palabra')
    print()
 
print('El corpus con compuestos largos necesita MAS tokens por palabra')
print('porque las palabras unicas son mas dificiles de comprimir:')
print('cada compuesto aparece pocas veces, asi que BPE no puede')
print('asignarle un token unico tan facilmente.')
print()
print('Es una ilustracion con dos corpus inventados, no una medida de')
print('tokenizadores reales; pero apunta en la misma direccion que el fenomeno')
print('real: idiomas de morfologia rica (aleman, finlandes, turco) gastan mas')
print('tokens por palabra en tokenizadores entrenados sobre todo en ingles.')
