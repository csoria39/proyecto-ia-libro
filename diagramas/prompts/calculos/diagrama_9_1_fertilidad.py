# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
"""Cifras del recuadro «Fertilidad» del Diagrama 9.1: cuántos tokens de GPT-2 gasta cada palabra o expresión.
Usa el tokenizador del Programa 8 bis, como el Ejercicio 16.1. Se ejecuta desde programas/ (necesita gpt2/)."""
import json, re
import numpy as np

CARPETA = 'gpt2'
fuente = open('programa_8bis_gpt2_real.py', encoding='utf-8').read()
exec('def bytes_a_unicode' + fuente.split('def bytes_a_unicode')[1].split('def generar')[0])

ejemplos = [('Inglés', ' intelligence'), ('Español', ' inteligencia'), ('Alemán', ' Künstliche Intelligenz'),
            ('Turco', ' Çekoslovakyalılaştıramadıklarımızdan'), ('Turco', ' evlerinizden'),
            ('Turco', ' gelemeyecekmişsiniz'), ('Español', ' supercalifragilistico'), ('Inglés', ' ChatGPT')]
for idioma, t in ejemplos:
    ids = codificar(t)
    piezas = [ID2TOK[i].replace('Ġ', '␣') for i in ids]
    print(f'{idioma:8s} {t.strip():38s} {len(ids):3d} tokens  {piezas}')
