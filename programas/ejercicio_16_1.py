# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
# EJERCICIO 16.1 — ¿Cuántos tokens gasta GPT-2 en cada idioma?
# Solo hace falta el tokenizador: se carga la parte del Programa 8 bis
# que va desde el tokenizador hasta los experimentos (sin leer los pesos).
import json, re
import numpy as np

CARPETA = 'gpt2'
fuente = open('programa_8bis_gpt2_real.py', encoding='utf-8').read()
exec('def bytes_a_unicode' + fuente.split('def bytes_a_unicode')[1].split('def generar')[0])

frases = {'inglés':  'Artificial intelligence learns from data and predicts the next word.',
          'español': 'La inteligencia artificial aprende de los datos y predice la siguiente palabra.',
          'alemán':  'Künstliche Intelligenz lernt aus Daten und sagt das nächste Wort voraus.',
          'griego':  'Η τεχνητή νοημοσύνη μαθαίνει από δεδομένα και προβλέπει την επόμενη λέξη.',
          'chino':   '人工智能从数据中学习并预测下一个词。'}
print(f"{'idioma':8s} {'caract.':>8s} {'palabras':>9s} {'tokens':>7s} {'tokens/palabra':>15s}")
for idioma, f in frases.items():
    ids = codificar(f)
    # en chino no hay espacios: contamos cada carácter (sin el punto) como una palabra
    palabras = len(f.split()) if idioma != 'chino' else len(f) - 1
    print(f'{idioma:8s} {len(f):8d} {palabras:9d} {len(ids):7d} {len(ids) / palabras:15.1f}')
for idioma in ['inglés', 'español']:
    print(f'\nPiezas del {idioma}:', [ID2TOK[i].replace("Ġ", "␣") for i in codificar(frases[idioma])])
