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
 
# El tiempo real no se deduce de aqui: depende del hardware y del modelo
print('Lo que cambia es la profundidad del camino secuencial, no un tiempo fijo:')
print(f'la RNN encadena {T} pasos que no pueden solaparse; el Transformer')
print('procesa las T posiciones a la vez, a cambio de una atencion que cuesta')
print('O(T^2) en calculo y memoria. Cuanto se gana en tiempo hay que medirlo')
print('en cada maquina, pero el paralelismo es la razon fundamental por la que')
print('se puede entrenar con billones de tokens.')
