# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
# EJERCICIO 16.2 — La lente de logits con frases propias
# Reutiliza la función lente() del Programa G-1.
import numpy as np

fuente = open('programa_G1_lente_logits.py', encoding='utf-8').read()
exec(fuente.split("\nlente('")[0])          # todo menos las llamadas finales

lente('The opposite of hot is', ' cold')
lente('Monday, Tuesday,', ' Wednesday')
