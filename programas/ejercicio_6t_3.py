# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
T = 1000   # longitud de la secuencia
 
print('=== PASOS SECUENCIALES: RNN vs TRANSFORMER ===\n')
 
print('RNN:')
print(f'  Pasos secuenciales necesarios: {T}')
print(f'  (cada paso DEBE esperar al anterior, sin excepcion)')
print(f'  Camino mas largo entre token 1 y token {T}: {T} pasos')
print()
 
print('Transformer (con self-attention):')
print(f'  Pasos secuenciales necesarios: 1 (todos los tokens en paralelo)')
print(f'  Camino mas largo entre cualquier par de tokens: 1 paso')
print(f'  (la atencion conecta directamente cualquier par)')
print()
 
# Estimacion de tiempo en una GPU con paralelismo masivo
tiempo_por_paso_ms = 0.1   # estimacion simplificada
tiempo_rnn         = T * tiempo_por_paso_ms
tiempo_transformer  = 1 * tiempo_por_paso_ms  # un solo paso paralelo
 
print(f'Tiempo estimado RNN:         {tiempo_rnn:.1f} ms')
print(f'Tiempo estimado Transformer: {tiempo_transformer:.1f} ms')
print(f'Speedup teorico: {tiempo_rnn/tiempo_transformer:.0f}x')
print()
print('En la practica el speedup real es menor (el Transformer tiene')
print('coste O(T^2) en memoria), pero el paralelismo es la razon')
print('fundamental por la que se puede entrenar con billones de tokens.')
