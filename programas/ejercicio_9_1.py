# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
# =======================================================
# EJERCICIO 9.1 — Entrenar el Nano-LLM con otro texto
# Reutiliza el Programa 8 tal cual: solo sustituye TEXTO,
# reduce PASOS y guarda los pesos con otro nombre.
# =======================================================
import numpy as np

NUEVO_TEXTO = """
python es el lenguaje de la inteligencia artificial.
las listas se crean con corchetes: [1, 2, 3].
los diccionarios usan llaves y pares clave valor.
las funciones se definen con la palabra reservada def.
los bucles for recorren secuencias de elementos.
las clases permiten crear objetos con metodos y atributos.
importa numpy para operaciones numericas eficientes.
las redes neuronales aprenden de los datos de entrenamiento.
"""

# Análisis del nuevo vocabulario (antes de entrenar)
vocab = sorted(set(NUEVO_TEXTO))
V = len(vocab)
print(f'Texto: {len(NUEVO_TEXTO)} caracteres | vocabulario: {V} caracteres únicos')
print(f'Pérdida inicial esperada: ln({V}) = {np.log(V):.4f}\n')

# Programa 8 con tres cambios: TEXTO, PASOS y nombre del archivo de pesos
fuente = open('programa_8_nano_llm.py', encoding='utf-8').read()
viejo = fuente.split('TEXTO = """')[1].split('"""')[0]
fuente = fuente.replace(viejo, NUEVO_TEXTO)
fuente = fuente.replace('PASOS  = 12000', 'PASOS  = 3000')
fuente = fuente.replace('nano_llm_pesos.npz', 'pesos_ejercicio_9_1.npz')
exec(fuente)
