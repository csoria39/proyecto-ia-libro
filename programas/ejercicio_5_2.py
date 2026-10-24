# SPDX-License-Identifier: MIT  (© 2026 Carlos Soria Martínez; ver LICENSE)
def contar_params(arquitectura):
    """Cuenta pesos + sesgos de una red densa."""
    total = 0
    for i in range(len(arquitectura) - 1):
        n_ent = arquitectura[i]
        n_sal = arquitectura[i+1]
        pesos  = n_ent * n_sal
        sesgos = n_sal
        total += pesos + sesgos
    return total

# Verificar con casos conocidos
casos = {
    'Una neurona (1 entrada)': [1, 1],
    'Neurona AND (2 entradas)': [2, 1],
    'MLP XOR (2-8-1)':         [2, 8, 1],
    'MLP mediano (10-64-32-1)':[10, 64, 32, 1],
    'FFN de un bloque de GPT-2 (768-3072-768)': [768, 3072, 768],
}

print(f"{'Arquitectura':>40}  {'Parámetros':>12}")
print('-' * 56)
for nombre, arq in casos.items():
    print(f'{nombre:>40}  {contar_params(arq):>12,}')

# Cálculo manual para verificar [2, 8, 1]:
# Capa 1: 2*8 + 8 = 16 + 8 = 24
# Capa 2: 8*1 + 1 =  8 + 1 =  9
# Total: 24 + 9 = 33, que coincide con el Programa 3 (8 ocultas): 2*8+8 + 8*1+1 = 33
