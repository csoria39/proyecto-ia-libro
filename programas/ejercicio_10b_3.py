# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
import numpy as np
 
np.random.seed(42)
 
def generar_benchmark(n, semilla):
    '''Genera un benchmark simulado de opcion multiple'''
    np.random.seed(semilla)
    return [{'id': i, 'respuesta_correcta': np.random.randint(4)} for i in range(n)]
 
benchmark_train = generar_benchmark(20, semilla=1)   # 'visto' en preentrenamiento
benchmark_nuevo = generar_benchmark(20, semilla=2)   # nunca visto
 
def modelo_memorizador(pregunta, benchmark_memorizado):
    '''Simula un modelo que memorizo las respuestas exactas'''
    for q in benchmark_memorizado:
        if q['id'] == pregunta['id']:
            return q['respuesta_correcta']  # responde perfecto si la vio
    return np.random.randint(4)  # si no la vio, responde al azar
 
def evaluar(modelo_fn, benchmark, memorizado):
    aciertos = sum(
        modelo_fn(q, memorizado) == q['respuesta_correcta']
        for q in benchmark
    )
    return aciertos / len(benchmark)
 
print('=== SIMULACION: OVERFITTING DE BENCHMARK ===\n')
 
acc_train = evaluar(modelo_memorizador, benchmark_train, benchmark_train)
acc_nuevo = evaluar(modelo_memorizador, benchmark_nuevo, benchmark_train)
 
print(f'Precision en benchmark CONTAMINADO (visto antes): {acc_train:.1%}')
print(f'Precision en benchmark NUEVO (nunca visto):       {acc_nuevo:.1%}')
print(f'Precision esperada al azar (4 opciones):           25.0%')
print()
print('El modelo "memorizador" parece perfecto (100%) en el benchmark')
print('contaminado, pero su rendimiento real (en preguntas nuevas)')
print('es indistinguible del azar. Esto es exactamente lo que ocurre')
print('cuando un LLM ha visto las respuestas del benchmark durante')
print('su preentrenamiento: la metrica miente sobre la capacidad real.')
