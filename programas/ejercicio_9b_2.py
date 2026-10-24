# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import numpy as np
 
def type_token_ratio(texto):
    palabras = texto.lower().split()
    if not palabras: return 0
    return len(set(palabras)) / len(palabras)
 
# Simulacion: con el Nano-LLM real, cargar params = dict(np.load('nano_llm_pesos.npz'))
def simular_generacion(tau, semilla, vocab_efectivo=15):
    '''Simula texto generado con cierta temperatura'''
    np.random.seed(semilla)
    palabras_base = ['la','red','aprende','de','los','datos','el',
                     'modelo','predice','token','siguiente','con',
                     'atencion','capas','neuronas']
    n_palabras = 30
    # A mayor tau, mas aleatoriedad en la eleccion (simulado)
    if tau < 0.5:
        # Baja temperatura: repite mucho un subconjunto pequeño
        pool = palabras_base[:4]
    elif tau < 1.0:
        pool = palabras_base[:10]
    else:
        pool = palabras_base
    return ' '.join(np.random.choice(pool, n_palabras))
 
print('=== DIVERSIDAD LEXICA SEGUN TEMPERATURA ===\n')
print(f"{'Temperatura':>12}  {'TTR medio':>10}  {'TTR std':>10}")
print('-' * 38)
 
for tau in [0.3, 0.7, 1.0, 1.3]:
    ratios = []
    for semilla in range(20):
        texto = simular_generacion(tau, semilla)
        ratios.append(type_token_ratio(texto))
    print(f'{tau:>12.1f}  {np.mean(ratios):>10.3f}  {np.std(ratios):>10.3f}')
 
print()
print('TTR mayor = mas variedad lexica = menos repeticion')
print('Temperaturas bajas producen TTR bajo (texto repetitivo)')
print('Temperaturas muy altas pueden producir TTR alto pero incoherente')
